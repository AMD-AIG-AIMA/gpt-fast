""" run evaluation on benchmark datasets
python evaluate.py --bench_name mt_bench --checkpoint_path checkpoints/meta-llama/llama-2-7b-chat-hf/llama-2-7b-chat --draft_checkpoint_path checkpoints/meta-llama/llama-2-7b-chat-hf/llama-2-7b-chat --speculate_k 5 --max_new_tokens 1024 \
"""
import json
import time
from pathlib import Path
from typing import Optional
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from fastchat.model import get_conversation_template
from tp import maybe_init_dist, _get_rank
from model import Transformer
from tokenizer import get_tokenizer
import itertools
from typing import Optional, Tuple, Union
from time_profiler import TimeProfiler
TimeProfiler.set_warm_up(5)

MODEL_TO_TEMPLATE = {
    "llama-2-7b-chat": "llama-2-chat",
    "llama-2-13b-chat": "llama-2-chat",
    "llama-2-70b-chat": "llama-2-chat",
    "mistral-7b-instruct": "mistral",
    "mixtral-8x7b-instruct": "mistral",
    "openchat-3.5": "openchat",
    "llama-3.1": "llama-3.1",
    "llama-3.2": "llama-3.2",
    "qwen2": "qwen2",
    # Add more mappings as needed
}
GPU_BANDWIDTH = {
    "MI325": 6e12,
    "MI300": 5.3e12,
    "MI250": 1.6e12,
    "H200": 4.8e12,
    "H100": 3.35e12,
    "A100": 2.039e12,
}
device_name = torch.cuda.get_device_name()
PEAK_BANDWIDTH = None
rank = _get_rank()
for d in GPU_BANDWIDTH.keys():
    if d in device_name:
        PEAK_BANDWIDTH = GPU_BANDWIDTH[d]
        if rank == 0 or rank is None:
            print(f"Found device {d} with peak bandwidth {PEAK_BANDWIDTH/1e12} TB/s")
        break
if PEAK_BANDWIDTH is None and (rank == 0 or rank is None):
    print("Device not found in the list of known devices. Please update the GPU_BANDWIDTH dictionary in evaluate.py")
    print("Using default peak bandwidth of 2 TB/s")
    PEAK_BANDWIDTH = 2e12

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_benchmark_data(bench_name):
    data_path = Path(f"./data/{bench_name}/question.jsonl")
    questions = []
    with open(data_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions

def get_model_template(model_name):
    for key, template in MODEL_TO_TEMPLATE.items():
        if key in model_name.lower():
            return template
    return "llama-2-chat"  # Default to llama-2-chat if no match found

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with sdpa_kernel([SDPBackend.MATH,SDPBackend.FLASH_ATTENTION]):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

def block_verify(target_logits, draft_probs, draft_tokens, speculate_k, device, sampling_kwargs):
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    if len(draft_probs.shape) > 2:
        draft_probs = draft_probs.squeeze(1)

    # Initialize p_i and h_i
    p = torch.ones(speculate_k+1, device=device)
    accept_length = 0
    eta = torch.rand(speculate_k, device=device)
    for i in range(speculate_k):
        q_target = target_probs[i]
        p_draft = draft_probs[i]
        draft_token = draft_tokens[i]

        # Calculate p_i, acceptance probability
        p[i+1] = torch.minimum(p[i] * q_target[draft_token] / p_draft[draft_token], torch.ones(1, device=device))    

        # Calculate residual distribution and h_i
        residual = torch.maximum(p[i+1] * q_target - p_draft, torch.zeros_like(q_target))
        residual_mass = torch.sum(residual)
        h_i = residual_mass / (residual_mass + 1 - p[i+1])

        # Acceptance step
        if eta[i] <= h_i:
            accept_length = i + 1
        else:
            continue

    if accept_length == speculate_k:
        # All draft tokens accepted, sample final token
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        return torch.cat([draft_tokens, last_token])
    else:
        # Sample from residual distribution
        p_draft = draft_probs[accept_length]
        q = target_probs[accept_length]
        residual = torch.maximum(p[accept_length] * q - p_draft, torch.zeros_like(q))
        residual = residual / residual.sum()
        next_token = multinomial_sample_one_no_sync(residual)
        return torch.cat([draft_tokens[:accept_length], next_token])

def token_verify(target_logits, draft_probs, draft_tokens, speculate_k, device, sampling_kwargs):
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    if len(draft_probs.shape) > 2:
        draft_probs = draft_probs.squeeze(1)
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        if p.shape[0] != q.shape[0]:
            q = q[:p.shape[0]]
            q /= q.sum(dim=-1, keepdim=True)
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    do_block_verify: bool = False,
    **sampling_kwargs
) -> torch.Tensor:
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=device)
    
    # Get draft tokens and probabilities
    with TimeProfiler("Speculate Decode", model_size=model_size(draft_model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
        profiler.set_tokens_processed(speculate_k)
        draft_tokens, draft_probs = decode_n_tokens(
            draft_model, 
            cur_token.view(1, -1), 
            orig_input_pos.clone(), 
            speculate_k, 
            **sampling_kwargs
        )

    draft_tokens = torch.cat(draft_tokens)
    if len(draft_tokens.shape) > 1:
        draft_tokens = draft_tokens.view(-1)
    # parallel inference on target model using draft tokens
    with TimeProfiler("Verification", model_size=model_size(model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
        profiler.set_tokens_processed(1)
        target_logits = model_forward(
            model,
            torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
            torch.arange(input_pos, input_pos + speculate_k + 1, device=device)
        )
    
    if not do_block_verify:
        with TimeProfiler("Token Verify"):
            verified_tokens = token_verify(target_logits, draft_probs, draft_tokens, speculate_k, device, sampling_kwargs)
    else:
        with TimeProfiler("Block Verify"):
            verified_tokens = block_verify(target_logits, draft_probs, draft_tokens, speculate_k, device, sampling_kwargs)
    
    # fill last token into draft model if all speculative tokens have been accepted
    if verified_tokens.shape[0] == speculate_k + 1:
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
    return verified_tokens

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    do_block_verify: bool = False,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with TimeProfiler("Prefill", model_size=model_size(model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
        profiler.set_tokens_processed(1)
        next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    if is_speculative:
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, do_block_verify, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[:,input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        new_tokens, new_probs = [], []
        cur_token = next_token.view(batch_size, -1)
        num_new_tokens = max_new_tokens - 1
        for i in range(num_new_tokens):
            with TimeProfiler("Vanilla Decoding", model_size=model_size(model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
                profiler.set_tokens_processed(1)
                with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]):
                    next_token, next_prob = decode_one_token(
                        model, cur_token, input_pos, **sampling_kwargs
                    )
                    input_pos += 1
                    new_tokens.append(next_token.clone())
                    callback(new_tokens[-1])
                    new_probs.append(next_prob.clone())
                    cur_token = next_token.clone()
        seq[:, T + 1:] = torch.cat(new_tokens, dim=-1)

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        if tokenizer.bos_id() is not None:
            tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()
        precision = 'int8'

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()
        precision = 'int4'

    if 'fp8' in str(checkpoint_path):
        print("Using fp8 weight-only quantization!")
        from quantize import WeightOnlyFP8QuantHandler
        simple_quantizer = WeightOnlyFP8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()
        precision = 'float8_e4m3fn'

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)


    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    print(f"Model dtype is: {model.output.weight.dtype}")
    return model.eval()

def model_size(model):
    if hasattr(model, "model_size_cache"):
        return model.model_size_cache
    else:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        # size_all_mb = (param_size + buffer_size) / 1024**2
        model.model_size_cache =  param_size + buffer_size
        return model.model_size_cache

def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

def main(
    bench_name: str,
    checkpoint_path: Path,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    compile: bool = False,
    compile_prefill: bool = False,
    num_questions: Optional[int] = None,
    warmup: int = 5,
    do_block_verify: bool = False,
):
    global print
    # Initialize distributed setup
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        print(f"Loading draft model from {draft_checkpoint_path}")
        draft_model = _load_model(draft_checkpoint_path, device, torch.bfloat16, use_tp=False)
    else:
        draft_model = None

    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(checkpoint_path.parent / "tokenizer.model", checkpoint_path)

    if compile:
        if is_speculative:
            global model_forward, logits_to_probs
            model_forward = torch.compile(model_forward, mode="max-autotune", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="max-autotune", fullgraph=True)

        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    questions = load_benchmark_data(bench_name)
    
    results = []
    speeds = []
    
    model_template = get_model_template(checkpoint_path.parent.name)
    
    system_message = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. ")
    
    # Warmup calls
    for i in range(warmup):
        question = questions[i]
        conv = get_conversation_template(model_template)
        conv.messages = []
        conv.set_system_message(system_message)

        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
            output, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                interactive=False,
                batch_size=1,
                draft_model=draft_model,
                speculate_k=speculate_k,
                callback=lambda x: x,
                do_block_verify=do_block_verify,
            )

    if num_questions is not None:
        questions = questions[:num_questions]
    for question in tqdm(questions):
        torch.manual_seed(0)

        conv = get_conversation_template(model_template)
        conv.messages = []
        conv.set_system_message(system_message)
        
        turns = []
        new_tokens = []
        wall_time = []

        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
            
            start_time = time.time()
            with TimeProfiler("generate"):
                output, metrics = generate(
                    model,
                    encoded,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    interactive=False,
                    batch_size=1,
                    draft_model=draft_model,
                    speculate_k=speculate_k,
                    callback=lambda x: x,
                    do_block_verify=do_block_verify,
                )
            end_time = time.time()
            
            output_ids = output[0][len(encoded):]
            if conv.stop_token_ids:
                stop_token_ids_index = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
                if stop_token_ids_index:
                    output_ids = output_ids[:stop_token_ids_index[0]]

            generated_text = tokenizer.decode(output_ids.tolist())
            
            if conv.stop_str:
                stop_str_index = generated_text.find(conv.stop_str)
                if stop_str_index != -1:
                    generated_text = generated_text[:stop_str_index]
            
            generated_text = generated_text.strip()
            
            walltime = end_time - start_time
            tokens_generated = len(output[0]) - len(encoded)
            speed = tokens_generated / walltime
            
            turns.append(generated_text)
            new_tokens.append(tokens_generated)
            wall_time.append(walltime)
            speeds.append(speed)
            
            conv.messages[-1][-1] = generated_text

        results.append({
            'question_id': question.get('question_id', len(results)),
            'turns': turns,
            'tokens_generated': new_tokens,
            'walltime': wall_time,
            'speed': speeds[-len(turns):],
            'accept_counts': metrics['accept_counts'] if is_speculative else None,
        })
    
    if rank == 0 or rank is None:
        output_file = Path(f"./results_{bench_name}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        total_tokens = sum(sum(result['tokens_generated']) for result in results)
        total_time = sum(sum(result['walltime']) for result in results)
        overall_tokens_per_second = total_tokens / total_time
        all_speeds = [speed for result in results for speed in result['speed']]
        mean_speed = sum(all_speeds) / len(all_speeds)
        max_speed = max(all_speeds)
        
        print(f"Overall generation speed: {overall_tokens_per_second:.2f} tokens/second")
        print(f"Mean generation speed: {mean_speed:.2f} tokens/second")
        print(f"Max generation speed: {max_speed:.2f} tokens/second")

        if is_speculative:
            counts_aggregated = [0] * (speculate_k + 1)
            for result in results:
                counts_aggregated = [count + result['accept_counts'][idx] for idx, count in enumerate(counts_aggregated)]
            total_counts = sum(counts_aggregated)
            acceptance_probs = [count / total_counts for count in counts_aggregated]
            print(f"Acceptance probs: {acceptance_probs}")
            print(f"Mean Accepted: {sum([(idx+1) * count for idx, count in enumerate(counts_aggregated[:-1])]) / sum(counts_aggregated[:-1]):.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on benchmark datasets.')
    parser.add_argument('--bench_name', type=str, required=True, help='Name of the benchmark (e.g., mt_bench, human_eval)')
    parser.add_argument('--checkpoint_path', type=Path, required=True, help='Path to the model checkpoint')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Path to the draft model checkpoint for speculative decoding')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill')
    parser.add_argument('--num_questions', type=int, default=None, help='Number of questions to evaluate')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup steps')
    parser.add_argument('--do_block_verify', action='store_true', help='Whether to verify with block acceptance probability')
    
    args = parser.parse_args()
    main(args.bench_name, args.checkpoint_path, args.max_new_tokens, args.temperature, args.top_k, args.device,
         args.draft_checkpoint_path, args.speculate_k, args.compile, args.compile_prefill, args.num_questions,
         args.warmup, args.do_block_verify)