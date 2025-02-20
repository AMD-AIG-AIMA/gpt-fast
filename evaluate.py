""" run evaluation on benchmark datasets
python evaluate.py --bench_name mt_bench --checkpoint_path checkpoints/meta-llama/llama-2-7b-chat-hf/llama-2-7b-chat --draft_checkpoint_path checkpoints/meta-llama/llama-2-7b-chat-hf/llama-2-7b-chat --speculate_k 5 --max_new_tokens 1024 \
"""
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from conversation import get_conversation_template
import itertools
from typing import Optional, Tuple, Union

from tp import maybe_init_dist, _get_rank
from model import Transformer
from tokenizer import get_tokenizer
from multimodal.vision_modules import VisionModule

from time_profiler import TimeProfiler
TimeProfiler.set_warm_up(0)


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

def load_benchmark_data(bench_name, bench_args=None):
    if bench_name in ["mt_bench", "human_eval"]:
        data_path = Path(f"./data/{bench_name}/question.jsonl")
        questions = []
        with open(data_path, 'r') as f:
            for line in f:
                questions.append(json.loads(line.strip()))
    elif bench_name in ['MMMU']:
        # A multimodal benchmark dataset
        from datasets import load_dataset
        categories = getattr(bench_args, 'categories', ['Design'])
        split = getattr(bench_args, 'split', 'dev')
        questions = []
        for cat in categories:
            dataset =  load_dataset("MMMU/MMMU", cat)[split]
            for i in range(len(dataset)):
                question = {}
                question['turns'] = [dataset[i]['question']]
                question['images'] = []
                for j in range(1, 8):
                    if dataset[i][f'image_{j}'] is not None:
                        question['images'].append(dataset[i][f'image_{j}'])
                    else:
                        break
                questions.append(question)
    else:
        raise ValueError(f"Unknown benchmark name: {bench_name}")
    return questions

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

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, embedded: Optional[torch.Tensor]=None,
            cross_states: Optional[torch.Tensor]=None, cross_attention_mask: Optional[Union[torch.Tensor, Dict]]=None, 
            **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    if cross_attention_mask is not None:
        cross_states = embedded.clone()
        embedded = None
    logits = model(x, input_pos, embedded=embedded, cross_states=cross_states, cross_attention_mask=cross_attention_mask)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, 
                     cross_attention_mask: Optional[Union[torch.Tensor, Dict]]=None,
                    **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos, cross_attention_mask=cross_attention_mask)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, 
                    num_new_tokens: int, callback=lambda _: _, 
                    cross_attention_mask: Optional[Union[torch.Tensor, Dict]]=None,
                    **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with sdpa_kernel([SDPBackend.MATH,SDPBackend.FLASH_ATTENTION]):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cross_attention_mask=cross_attention_mask, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

    return new_tokens, new_probs


def model_forward(model, x, input_pos, cross_attention_mask: Optional[Union[torch.Tensor, Dict]]=None):
    return model(x, input_pos, cross_attention_mask=cross_attention_mask)

def cross_attention_mask_update(cross_attention_mask, num_tokens=1):
    if cross_attention_mask is not None:
        cross_attention_mask = {
            'mask': cross_attention_mask['mask'][:,:,-1,:].repeat(1, 1, num_tokens, 1),
            'out_mask': cross_attention_mask['out_mask'][:,:,-1,:].repeat(1, 1, num_tokens, 1)
        }
    return cross_attention_mask
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
    draf_input_pos: Optional[int] = None,
    cross_attention_mask: Optional[Union[torch.Tensor, Dict]]=None,
    draft_cross_attention_mask: Optional[Union[torch.Tensor, Dict]]=None,
    **sampling_kwargs
) -> torch.Tensor:
    device = cur_token.device
    if draf_input_pos is None:
        draf_input_pos = input_pos
    orig_input_pos = torch.tensor([draf_input_pos], dtype=torch.int64, device=device)
    
    # Get draft tokens and probabilities
    with TimeProfiler("Speculate Decode", model_size=model_size(draft_model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
        profiler.set_tokens_processed(speculate_k)
        draft_tokens, draft_probs = decode_n_tokens(
            draft_model, 
            cur_token.view(1, -1), 
            orig_input_pos.clone(), 
            speculate_k, 
            cross_attention_mask=draft_cross_attention_mask,
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
            torch.arange(input_pos, input_pos + speculate_k + 1, device=device),
            cross_attention_mask=cross_attention_mask
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
            cross_attention_mask=draft_cross_attention_mask
        )
    return verified_tokens

def calculate_sequence_lengths(
    prompt: torch.Tensor,
    max_new_tokens: int,
    embedded: Optional[torch.Tensor],
    draft_encoded: Optional[torch.Tensor],
    speculate_k: Optional[int],
    is_speculative: bool,
    interactive: bool,
    model_block_size: int,
    cross_attention_mask: Optional[Union[torch.Tensor, Dict]] = None,
) -> dict:
    """Calculate various sequence lengths needed for generation.
    
    Args:
        prompt: Input token tensor
        max_new_tokens: Maximum number of tokens to generate
        embedded: Optional embedded tensor for multimodal models. Can be:
            - None: Text-only model
            - [B, L, E] tensor: Direct token embeddings
            - [P, L, E] tensor: Cross-attention states where P is patch size
        draft_encoded: Optional encoded tensor for text-only draft models
        speculate_k: Number of tokens to speculate (if using speculative decoding)
        is_speculative: Whether using speculative decoding
        interactive: Whether in interactive mode
        model_block_size: Maximum sequence length supported by model
        cross_attention_mask: Optional cross attention mask to determine if using cross attention
    
    Returns:
        Dictionary containing calculated lengths:
        - input_text_length: Length of input text tokens
        - input_embed_length: Length of input embeddings (same as text length for cross-attention case)
        - text_seq_length: Total text sequence length
        - embed_seq_length: Total embedding sequence length
        - draft_text_seq_length: Sequence length for draft model
        - is_cross_attention: Whether using cross attention states
    """
    input_text_length = prompt.size(-1)
    
    # Determine if we're using cross attention states or direct embeddings
    is_cross_attention = cross_attention_mask is not None and embedded is not None
    multimodal = embedded is not None
    
    # For cross attention case, input_embed_length is same as text length
    # For direct embedding case, use embedding length
    if multimodal and not is_cross_attention:
        input_embed_length = embedded.size(1)  # Use L from [B, L, E]
    else:
        input_embed_length = input_text_length  # Same as text length for text-only or cross-attention
    
    # Calculate base sequence lengths
    text_seq_length = input_text_length + max_new_tokens
    
    # For cross attention case, embed_seq_length follows text length
    # For direct embedding case, use embedding length + new tokens
    if multimodal and not is_cross_attention:
        embed_seq_length = input_embed_length + max_new_tokens
    else:
        embed_seq_length = text_seq_length
    
    # Adjust for interactive mode or model limits
    if interactive:
        embed_seq_length = 350
    else:
        embed_seq_length = min(embed_seq_length, model_block_size)
    
    # Add extra space for speculative decoding if needed
    if is_speculative:
        embed_seq_length += speculate_k + 1
        text_seq_length += speculate_k + 1
        
    # Calculate draft model sequence length if needed
    draft_text_seq_length = (
        draft_encoded.size(-1) + speculate_k + 1 + max_new_tokens 
        if draft_encoded is not None 
        else text_seq_length
    )
    
    # Verify text-only models have matching lengths
    if not multimodal:
        assert text_seq_length == embed_seq_length, "Text-only model should have the same embed_seq_length as text_seq_length"
        
    return {
        "input_text_length": input_text_length,
        "input_embed_length": input_embed_length,
        "text_seq_length": text_seq_length,
        "embed_seq_length": embed_seq_length,
        "draft_text_seq_length": draft_text_seq_length,
        "is_cross_attention": is_cross_attention
    }

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
    embedded: Optional[torch.Tensor] = None,
    draft_encoded: Optional[torch.Tensor] = None,
    draft_embedded: Optional[torch.Tensor] = None,
    cross_attention_mask: Optional[Union[torch.Tensor, Dict]]=None,
    draft_cross_attention_mask: Optional[Union[torch.Tensor, Dict]]=None,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    is_speculative = draft_model is not None
    multimodal = embedded is not None
    draft_multimodal = draft_embedded is not None
    if draft_multimodal and draft_encoded is not None:
        raise ValueError("Draft model is multimodal, but `draft_encoded` is also provided! \
                          `draft_encoded` is only used for multimodal target and text-only draft models.")

    # Calculate sequence lengths
    lengths = calculate_sequence_lengths(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        embedded=embedded,
        draft_encoded=draft_encoded,
        speculate_k=speculate_k,
        is_speculative=is_speculative,
        interactive=interactive,
        model_block_size=model.config.block_size,
        cross_attention_mask=cross_attention_mask
    )
    
    device, dtype = prompt.device, prompt.dtype
    
    # Setup model caches
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=lengths["embed_seq_length"])
        if is_speculative and draft_model is not model:
            if draft_embedded is not None:
                draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=lengths["embed_seq_length"])
            else:
                draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=lengths["draft_text_seq_length"])

    # Initialize sequence tensor
    seq = torch.empty(batch_size, lengths["text_seq_length"], dtype=dtype, device=device)
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    seq[:, :lengths["input_text_length"]] = prompt
    
    input_pos = torch.arange(0, lengths["input_embed_length"], device=device)
    draft_input_pos = torch.arange(0, draft_encoded.size(-1), device=device) if draft_encoded is not None else input_pos

    with TimeProfiler("Prefill", model_size=model_size(model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
        profiler.set_tokens_processed(1)
        if multimodal:
            next_token = prefill(model, prompt.view(batch_size, -1), input_pos, embedded, cross_attention_mask=cross_attention_mask, **sampling_kwargs).clone()
        else:
            next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
        if is_speculative:
            if draft_multimodal:
                prefill(draft_model, prompt.view(batch_size, -1), input_pos, embedded, cross_attention_mask=draft_cross_attention_mask, **sampling_kwargs)
            elif multimodal:
                # Target multimodal, draft text only
                prefill(draft_model, draft_encoded.view(batch_size, -1), draft_input_pos, **sampling_kwargs)
            else:
                prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    seq[:, lengths["input_text_length"]] = next_token.squeeze()
    # Update cross attention mask for decoding stage
    cross_attention_mask = cross_attention_mask_update(cross_attention_mask, speculate_k + 1 if is_speculative else 1)
    draft_cross_attention_mask = cross_attention_mask_update(draft_cross_attention_mask, 1)

    input_pos = torch.tensor([lengths["input_embed_length"]], device=device, dtype=torch.int)
    
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        draft_input_pos = draft_encoded.size(-1) if draft_encoded is not None else input_pos
        max_pos = lengths["embed_seq_length"] - 1 - (speculate_k + 1)
        while input_pos < max_pos:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, do_block_verify, 
                draft_input_pos, cross_attention_mask=cross_attention_mask,
                draft_cross_attention_mask=draft_cross_attention_mask, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(lengths["embed_seq_length"] - input_pos - 1, len(next_tokens))
            if not multimodal:
                seq[:,input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            else:
                text_input_pos = input_pos - (lengths["input_embed_length"] - lengths["input_text_length"]) + 1
                seq[:, text_input_pos : text_input_pos + num_added] = next_tokens[: num_added]
            # for i in next_tokens[: num_added,]:
            #     callback(i)
            input_pos = input_pos + num_added
            draft_input_pos = draft_input_pos + num_added
            next_token = next_tokens[-1]
        text_input_pos = input_pos - (lengths["input_embed_length"] - lengths["input_text_length"]) + 1
        seq = seq[:, :text_input_pos]
    else:
        new_tokens, new_probs = [], []
        cur_token = next_token.view(batch_size, -1)
        num_new_tokens = max_new_tokens - 1
        for i in range(num_new_tokens):
            with TimeProfiler("Vanilla Decoding", model_size=model_size(model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
                profiler.set_tokens_processed(1)
                with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]):
                    next_token, next_prob = decode_one_token(
                        model, cur_token, input_pos, 
                        cross_attention_mask=cross_attention_mask,
                        **sampling_kwargs
                    )
                    input_pos += 1
                    new_tokens.append(next_token.clone())
                    # callback(new_tokens[-1])
                    new_probs.append(next_prob.clone())
                    cur_token = next_token.clone()
        end = len(new_tokens) + lengths["input_text_length"] +1
        seq[:, lengths["input_text_length"] + 1:end] = torch.cat(new_tokens, dim=-1)

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
    precision = torch.float16
    is_speculative = draft_checkpoint_path is not None
    
    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)
    model.requires_grad_(False) 
    multimodal = getattr(model.config, "mm_config", None) is not None
    if multimodal:
        torch.set_default_dtype(precision)
        torch.set_default_device(device)
        vision_checkpoints = checkpoint_path.parent / "vision_modules.pth"
        vision_modules = VisionModule.from_name(checkpoint_path.parent.name, 
                                                config=model.config.mm_config,
                                                checkpoint_path=vision_checkpoints,
                                                dtype=precision,
                                                device=device)
        vision_modules.eval_mode()
    else:
        vision_modules = None

    if is_speculative:
        print(f"Loading draft model from {draft_checkpoint_path}")
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp=False)
        draft_multimodal =  getattr(draft_model.config, "mm_config", None) is not None
        if draft_multimodal:
            draft_vision_checkpoints = str(draft_checkpoint_path.parent / "vision_modules.pth")
            draft_vision_modules = VisionModule.from_name(draft_checkpoint_path.parent.name, 
                                                        config=draft_model.config.mm_config, 
                                                        checkpoint_path=draft_vision_checkpoints,
                                                        dtype=precision,
                                                        device=device)
            draft_vision_modules.eval_mode()
        else:
            draft_vision_modules = None
    else:
        draft_model = None
        draft_vision_modules = None

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
    
    conv = get_conversation_template(checkpoint_path)
    
    system_message = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. ")
    
    # Warmup calls
    # TODO: For multimodal models, we need to warmup the vision modules as well, if we compile them
    for i in range(warmup):
        question = questions[i]
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
            
            if not multimodal:
                encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
                embedded, draft_encoded, draft_embedded, cross_attention_mask, draft_cross_attention_mask = None, None, None, None, None
            else:
                with TimeProfiler("Embedding"):
                    with torch.inference_mode():
                        encoded, embedded = vision_modules(
                            prompt=prompt, tokenizer=tokenizer, images=question["images"],
                            embed_tokens=model.tok_embeddings, 
                        )
                        cross_attention_mask = getattr(vision_modules, "cross_attention_mask", None)
                        if is_speculative and draft_multimodal:
                            _, draft_embedded = draft_vision_modules(
                                prompt=prompt, tokenizer=tokenizer, images=question["images"],
                                embed_tokens=draft_model.tok_embeddings, 
                            )
                            draft_encoded = None
                            draft_cross_attention_mask = getattr(draft_vision_modules, "cross_attention_mask", None)
                        elif is_speculative and not draft_multimodal:
                            # Target model is multimodal, and draft model is text only -> encoding would be different if <image> token is present
                            draft_encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
                            draft_embedded, draft_cross_attention_mask = None, None
                        else:
                            draft_encoded, draft_embedded, draft_cross_attention_mask = None, None, None
                        encoded = encoded.squeeze(0)
            
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
                    embedded=embedded,
                    draft_encoded=draft_encoded,
                    draft_embedded=draft_embedded,
                    cross_attention_mask=cross_attention_mask,
                    draft_cross_attention_mask=draft_cross_attention_mask
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
            print(f"Mean Accepted: {sum([(idx) * count for idx, count in enumerate(counts_aggregated)]) / sum(counts_aggregated):.2f}")

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
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args.bench_name, args.checkpoint_path, args.max_new_tokens, args.temperature, args.top_k, args.device,
         args.draft_checkpoint_path, args.speculate_k, args.compile, args.compile_prefill, args.num_questions,
         args.warmup, args.do_block_verify)