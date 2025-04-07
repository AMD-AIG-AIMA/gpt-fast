""" run evaluation on benchmark datasets
python evaluate.py --bench_name mt_bench --checkpoint_path checkpoints/meta-llama/llama-2-7b-chat-hf/llama-2-7b-chat --draft_checkpoint_path checkpoints/meta-llama/llama-2-7b-chat-hf/llama-2-7b-chat --speculate_k 5 --max_new_tokens 1024 \
"""
import json
import time
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
import copy

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import BatchFeature 

from conversation import get_conversation_template
from tp import init_dist, _get_rank, broadcast, is_dist_initialized
from model import Transformer
from tokenizer import get_tokenizer
from multimodal.vision_modules import VisionModule
from time_profiler import TimeProfiler
from utils import (load_benchmark_data, 
                   calculate_sequence_lengths,
                   load_model,
                   model_size,
                )


TimeProfiler.set_warm_up(5)
torch._logging.set_logs(graph_breaks=True, recompiles=True)



GPU_BANDWIDTH = {
    "MI325": 6e12,
    "MI300": 5.3e12,
    "MI250": 1.6e12,
    "H200": 4.8e12,
    "H100": 3.35e12,
    "A100": 2.039e12,
}

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "int8": torch.int8,
    # "int4": torch.int4,
    "fp8": torch.float8_e4m3fn,
    "bf16": torch.bfloat16,    
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
            cross_states: Optional[torch.Tensor]=None, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    if getattr(model, "cross_attention_mask", None) is not None and embedded is not None:
        cross_states = embedded.clone()
        embedded = None
    logits = model(x, input_pos, embedded=embedded, cross_states=cross_states)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, 
                    **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, 
                    num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with sdpa_kernel([SDPBackend.MATH,SDPBackend.FLASH_ATTENTION]):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(next_token)
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

def draft_model_forward(model, x, input_pos):
    return model(x, input_pos)

# Helper function to detect if an object is a generator
def is_generator(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__next__') and callable(obj.__next__)

def token_verify(target_logits, draft_probs, draft_tokens, speculate_k, device, sampling_kwargs):
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs).to(device)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    if len(draft_probs.shape) > 2:
        draft_probs = draft_probs.squeeze(1)
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    # if is_dist_initialized():
    #     broadcast(p, 0)
    #     broadcast(q, 0)
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rand_vals = torch.rand_like(accept_draft_prob)
    # if is_dist_initialized():
    #     broadcast(rand_vals, 0)
    rejected_locations = (rand_vals > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k
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
    draf_input_pos: Optional[int] = None,
    **sampling_kwargs
) -> torch.Tensor:
    device = cur_token.device
    draft_device = draft_model._device  # Get draft model's device
    if draf_input_pos is None:
        draf_input_pos = input_pos
    orig_input_pos = torch.tensor([draf_input_pos], dtype=torch.int64, device=draft_device)
    # Get draft tokens and probabilities
    with TimeProfiler("Speculate Decode", model_size=model_size(draft_model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
        profiler.set_tokens_processed(speculate_k)
        draft_tokens, draft_probs = decode_n_tokens(
            draft_model, 
            cur_token.view(1, -1).to(draft_device), 
            orig_input_pos.clone(), 
            speculate_k,
            **sampling_kwargs
        )
    
    # Move draft outputs to target model device
    draft_tokens = torch.cat(draft_tokens).to(device)
    if len(draft_tokens.shape) > 1:
        draft_tokens = draft_tokens.view(-1)
    
    # parallel inference on target model using draft tokens
    with TimeProfiler("Verification", model_size=model_size(model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
        profiler.set_tokens_processed(1)
        # TODO: check if barrier is required here
        target_logits = model_forward(
            model,
            torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
            torch.arange(input_pos, input_pos + speculate_k + 1, device=device),
        )
    
    # One additional token to save the length of the verified tokens
    verified_tokens = -1 * torch.ones(speculate_k + 2, device=device, dtype=draft_tokens.dtype)
    with TimeProfiler("Token Verify"):
        new_tokens = token_verify(target_logits, draft_probs, draft_tokens, speculate_k, device, sampling_kwargs)
    if is_dist_initialized():
        # To avoid numerical issues and mismatch in the verified tokens of each node
        verified_tokens[:len(new_tokens)] = new_tokens
        verified_tokens[-1] = len(new_tokens)
        broadcast(verified_tokens, 0)
        verified_tokens = verified_tokens[:verified_tokens[-1]]
    else:
        verified_tokens = new_tokens
    
    # fill last token into draft model if all speculative tokens have been accepted
    if verified_tokens.shape[0] == speculate_k + 1:
        draft_model_forward(
                draft_model,
                draft_tokens[-1].view(1, -1).to(draft_device),
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
    interactive: bool, # TODO: should be merged with streaming
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    embedded: Optional[torch.Tensor] = None,
    draft_encoded: Optional[torch.Tensor] = None,
    draft_prompt: Optional[torch.Tensor] = None,
    draft_embedded: Optional[torch.Tensor] = None,
    max_cache_size: Optional[int] = None,
    draft_max_cache_size: Optional[int] = None,
    cross_attention_seq_length: Optional[int] = None,
    start_pos: Optional[dict] = {"target": 0, "draft": 0},
    streaming: bool = False,
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
        draft_embedded=draft_embedded,
        speculate_k=speculate_k,
        is_speculative=is_speculative,
        interactive=interactive,
        model_block_size=model.config.block_size,
        cross_attention_mask=model.cross_attention_mask,
        draft_cross_attention_mask=draft_model.cross_attention_mask if draft_model is not None else None,
        max_cache_size=max_cache_size,
        draft_max_cache_size=draft_max_cache_size,
        cross_attention_seq_length=cross_attention_seq_length,
        
    )

    draft_prompt = draft_prompt.to(draft_model._device) if draft_prompt is not None else None
    device, dtype = prompt.device, prompt.dtype
    # Setup model caches
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_cache_size=lengths["max_cache_size"], prompt=prompt,
                           cross_attention_seq_length=lengths["cross_attention_seq_length"])
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=batch_size, max_cache_size=lengths["draft_max_cache_size"], prompt=draft_prompt,
                                     cross_attention_seq_length=lengths["cross_attention_seq_length"])

    # Initialize sequence tensor
    seq = torch.empty(batch_size, lengths["text_seq_length"], dtype=dtype, device=device)
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    seq[:, :lengths["input_text_length"]] = prompt
    
    input_pos = torch.arange(start_pos["target"], start_pos["target"] + lengths["input_embed_length"], device=device)
    if is_speculative:
        draft_input_pos = torch.arange(start_pos["draft"], start_pos["draft"] + lengths["draft_input_embed_length"], device=draft_model._device)

    with TimeProfiler("Prefill", model_size=model_size(model), peak_bandwidth=PEAK_BANDWIDTH) as profiler:
        profiler.set_tokens_processed(1)
        if multimodal:
            next_token = prefill(model, prompt.view(batch_size, -1), input_pos, embedded, **sampling_kwargs).clone()
        else:
            next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
        if is_speculative:
            draft_device = draft_model._device
            if draft_multimodal:
                prefill(draft_model, draft_prompt.view(batch_size, -1).to(draft_device),
                        draft_input_pos.to(draft_device), draft_embedded, **sampling_kwargs)
            elif multimodal:
                # Target multimodal, draft text only
                prefill(draft_model, draft_encoded.view(batch_size, -1), draft_input_pos, **sampling_kwargs)
            else:
                prefill(draft_model, prompt.view(batch_size, -1).to(draft_device),
                        draft_input_pos.to(draft_device), **sampling_kwargs)
    if streaming:
        yield next_token
    prefill_time, prefill_tokens_per_second = profiler.get_last_call_stats()
    seq[:, lengths["input_text_length"]] = next_token.squeeze()
    input_pos = torch.tensor([lengths["input_embed_length"]], device=device, dtype=torch.int)
    
    accept_counts = [0] * (speculate_k + 1)
    acceptance_lengths = []

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        draft_input_pos = lengths["draft_input_embed_length"]
        max_pos = lengths["embed_seq_length"] - 1 - (speculate_k + 1)
        while input_pos < max_pos:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, 
                draft_input_pos, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            acceptance_lengths.append(len(next_tokens) - 1)
            num_added = min(lengths["embed_seq_length"] - input_pos - 1, len(next_tokens))
            if not multimodal:
                seq[:,input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            else:
                text_input_pos = input_pos - (lengths["input_embed_length"] - lengths["input_text_length"]) + 1
                seq[:, text_input_pos : text_input_pos + num_added] = next_tokens[: num_added]
            input_pos = input_pos + num_added
            draft_input_pos = draft_input_pos + num_added
            next_token = next_tokens[-1]
            should_stop = callback(next_tokens[: num_added,])
            if should_stop:
                break
            if streaming:
                yield next_tokens[: num_added]
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
                        **sampling_kwargs
                    )
                    input_pos += 1
                    new_tokens.append(next_token.clone())
                    should_stop = callback(next_token)
                    if should_stop:
                        break
                    if streaming:
                        yield next_token
                    new_probs.append(next_prob.clone())
                    cur_token = next_token.clone()
        end = len(new_tokens) + lengths["input_text_length"] +1
        seq[:, lengths["input_text_length"] + 1:end] = torch.cat(new_tokens, dim=-1)
        seq = seq[:, :end]

    generate_stats = {
        'accept_counts': accept_counts,
        'prefill_time': prefill_time,
        'acceptance_lengths': acceptance_lengths,
        'target_input_pos': input_pos.item(),
        'draft_input_pos': draft_input_pos.item() if is_speculative else 0
    }
    
    # Only return sequence and stats if not streaming (if callback is a generator, we've been yielding)
    return seq, generate_stats


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        if tokenizer.bos_id() is not None:
            tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def process_one_question(question, model, tokenizer, conv, max_new_tokens, temperature, top_k,
                     draft_model, speculate_k, device, vision_modules=None,
                     draft_vision_modules=None, max_cache_size=None, draft_max_cache_size=None, 
                     cross_attention_seq_length=None, mm_prune_method=None, mm_prune_ratio=0.0, 
                     turn=0, start_pos={"target": 0, "draft": 0}, streaming=False):
    is_speculative = draft_model is not None
    multimodal = vision_modules is not None
    draft_multimodal = draft_vision_modules is not None
    result = {}

    stop_token_ids_set = set(conv.stop_token_ids)
    stop_token_ids_tensor = torch.tensor(conv.stop_token_ids, device=device)
    
    # Create an optimized callback function that handles both single tokens and batches
    def token_callback(tokens):
        # Handle the case where tokens is a single token tensor
        if tokens.numel() == 1:
            if tokens.item() in stop_token_ids_set:
                return True
        else:
            # Handle batch of tokens with vectorized operation
            if conv.stop_token_ids:
                # Check if any token in the batch is a stop token
                is_stop = torch.isin(tokens, stop_token_ids_tensor.to(tokens.device))
                should_stop = is_stop.any().item()
                return should_stop
        return False

    # Add the user's message to the conversation
    conv.append_message(conv.roles[0], question['turns'][turn])
    
    # Get the prompt for processing - get only the latest turn
    if len(conv.messages) == 1:
        prompt = conv.get_prompt_for_generation()
    else:
        # For multiturn conversation, we only need the last conversation
        full_prompt = conv.get_prompt_for_generation()
        temp_conv = copy.deepcopy(conv)
        temp_conv.messages = conv.messages[:-1]
        past_prompt = temp_conv.get_prompt()
        prompt = full_prompt[len(past_prompt):]
    
    images = question.get('images',[])
    if not multimodal or len(images)==0:
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
        embedded, draft_encoded, draft_embedded, draft_prompt = None, None, None, None
    else:
        with TimeProfiler("Embedding"):
            with torch.inference_mode():
                encoded, embedded = vision_modules(
                    prompt=prompt, tokenizer=tokenizer, images=images,
                    embed_tokens=model.tok_embeddings, 
                )
                if isinstance(encoded, BatchFeature):
                    if 'image_grid_thw' in encoded:
                        model.image_grid_thw = encoded['image_grid_thw']
                    encoded = encoded['input_ids']
                if len(conv.messages)<2:
                    model.cross_attention_mask = getattr(vision_modules, "cross_attention_masks", {}).get('cross_attention_mask', None)
                    model.cross_attention_mask_out = getattr(vision_modules, "cross_attention_masks", {}).get('cross_attention_mask_out', None)
                if is_speculative and draft_multimodal:
                    # Set temporary default device for vision modules loading
                    if model._device != draft_model._device:
                        torch.set_default_device(draft_model._device)
                    draft_prompt, draft_embedded = draft_vision_modules(
                        prompt=prompt, tokenizer=tokenizer, images=images,
                        embed_tokens=draft_model.tok_embeddings, 
                        prune_method=mm_prune_method, prune_ratio=mm_prune_ratio,
                    )
                    # Reset default device
                    torch.set_default_device(device)
                    if isinstance(draft_prompt, BatchFeature):
                        if 'image_grid_thw' in draft_prompt:
                            draft_model.image_grid_thw = draft_prompt['image_grid_thw']
                        draft_prompt = draft_prompt['input_ids']
                    draft_encoded = None
                    if len(conv.messages)<2:
                        draft_model.cross_attention_mask = getattr(draft_vision_modules, "cross_attention_masks", {}).get('cross_attention_mask', None)  
                        draft_model.cross_attention_mask_out = getattr(draft_vision_modules, "cross_attention_masks", {}).get('cross_attention_mask_out', None)
                elif is_speculative and not draft_multimodal:
                    # Target model is multimodal, and draft model is text only -> encoding would be different if <image> token is present
                    draft_encoded = encode_tokens(tokenizer, prompt, bos=True, device=draft_model._device)
                    draft_embedded = None
                    draft_prompt = None
                else:
                    draft_encoded, draft_embedded, draft_prompt = None, None, None
                encoded = encoded.squeeze(0)
    
    start_time = time.time()
    

    with TimeProfiler("generate"):
        # Store generated tokens for post-streaming analysis
        all_streamed_tokens = []
        output = None
        metrics = None
        
        # Run generation with modified handling to capture both streamed tokens and final output
        generator = generate(
            model,
            encoded,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            interactive=False,
            batch_size=1,
            draft_model=draft_model,
            speculate_k=speculate_k,
            callback=token_callback,
            embedded=embedded,
            draft_prompt=draft_prompt,
            draft_encoded=draft_encoded,
            draft_embedded=draft_embedded,
            max_cache_size=max_cache_size,
            draft_max_cache_size=draft_max_cache_size,
            cross_attention_seq_length=cross_attention_seq_length,
            start_pos=start_pos,
            streaming=streaming,
        )
        try:
            if streaming:
                # Streaming behavior
                while True:
                    tokens = next(generator)
                    all_streamed_tokens.append(tokens)
                    yield tokens
            else:
                while True:
                    # generate is a generator, so we need to call next to get the final output
                    _ = next(generator)
        except StopIteration as e:
            output, metrics = e.value
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

    result['walltime'] = end_time - start_time - metrics['prefill_time']
    result['tokens_generated'] = len(output[0]) - len(encoded)
    result['speed'] = (result['tokens_generated'] - 1)/ result['walltime']
    result['accept_counts'] = metrics['accept_counts']
    result['generated_text'] = generated_text
    result['prefill_time'] = metrics['prefill_time']
    result['acceptance_lengths'] = metrics['acceptance_lengths']
    start_pos['target'] += metrics['target_input_pos']
    if is_speculative:
        start_pos['draft'] += metrics['draft_input_pos']
    return start_pos, result

def process_questions(questions, model, tokenizer, conv, system_message, max_new_tokens, temperature, top_k,
                     draft_model, speculate_k, device, vision_modules=None,
                     draft_vision_modules=None, max_cache_size=None, draft_max_cache_size=None, cross_attention_seq_length=None,
                     collect_metrics=False, eval_info={}, mm_prune_method=None, mm_prune_ratio=0.0):
    """Process a list of questions through the model and return results.
    
    Args:
        questions: List of questions to process
        model: Main model for generation
        tokenizer: Tokenizer instance
        conv: Conversation template
        system_message: System message to use
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Top-k for sampling
        draft_model: Optional draft model for speculative decoding
        speculate_k: Number of tokens to speculate
        multimodal: Whether using multimodal model
        vision_modules: Optional vision modules for multimodal
        draft_vision_modules: Optional draft vision modules
        max_cache_size: Maximum sequence length used for KVCache
        draft_max_cache_size: Maximum sequence length used for draft KVCache
        cross_attention_seq_length: Maximum sequence length used for KV-cache of cross_attention modules
        collect_metrics: Whether to collect and return metrics
        eval_info: Evaluation information to be saved with the log
        mm_prune_method: Method for pruning multimodal tokens ('none', 'random', 'structured')
        mm_prune_ratio: Ratio of multimodal tokens to prune (0.0 to 1.0)
        
    Returns:
        List of results if collect_metrics=True, otherwise None
    """
    is_speculative = draft_model is not None
    results = [] if collect_metrics else None
    speeds = [] if collect_metrics else None

    for question in tqdm(questions) if collect_metrics else questions:
        conv.messages = []
        conv.set_system_message(system_message)
        
        turns = []
        new_tokens = []
        wall_time = []
        prefill_time = []
        accept_counts = []
        start_pos = {"target": 0, "draft": 0}
        for j in range(len(question["turns"])):
            generator = process_one_question(question, model, tokenizer, conv, max_new_tokens, temperature, top_k,
                        draft_model, speculate_k, device, vision_modules,
                        draft_vision_modules, max_cache_size, draft_max_cache_size, cross_attention_seq_length,
                        mm_prune_method, mm_prune_ratio, j, start_pos)
            try:
                while True:
                    _ = next(generator)
            except StopIteration as e:
                start_pos, result = e.value
            turns.append(result['generated_text'])
            new_tokens.append(result['tokens_generated'])
            wall_time.append(result['walltime'])
            prefill_time.append(result['prefill_time'])
            accept_counts.append(result['accept_counts'])
            speeds.append(result['speed'])

        # Clear cache after processing this question
        model.clear_cache()
        if is_speculative:
            draft_model.clear_cache()
        if collect_metrics:
            results.append({
              'question_id': question.get('question_id', len(results)),
              'turns': turns,
              'tokens_generated': new_tokens,
              'walltime': wall_time,
              'prefill_time': prefill_time,
              'speed': speeds[-len(turns):],
              'accept_counts': accept_counts if is_speculative else None,
              'category': question.get('cat', None),
              'model': eval_info.get('model', ''),
              'draft_model': eval_info.get('draft_model', ''),
              'benchmark': eval_info.get('benchmark', ''),
              'speculative k': eval_info.get('speculative k', ''),
              'compile': eval_info.get('compile', ''),
              'compile_prefill': eval_info.get('compile_prefill', ''),
              'temperature': eval_info.get('temperature', ''),
              'top_k': eval_info.get('top_k', ''),
              'use_tp': eval_info.get('use_tp', ''),
              'max_cache_size': max_cache_size,
              'draft_max_cache_size': draft_max_cache_size,
              'mm_prune_method': mm_prune_method,
              'mm_prune_ratio': mm_prune_ratio,
            })
    return results

def main(
    bench_name: str,
    checkpoint_path: Path,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: str = 'fp16',
    draft_checkpoint_path: Optional[Path] = None,
    draft_device: Optional[str] = None,
    speculate_k: int = 5,
    compile: bool = False,
    compile_prefill: bool = False,
    num_questions: Optional[int] = None,
    warmup: int = 5,
    max_cache_size: int = None,
    draft_max_cache_size: int = None,
    cross_attention_seq_length: int = None,
    mm_prune_method: str = None,
    mm_prune_ratio: float = 0.0,
    streaming: bool = False,
):
    global print
    # Initialize distributed setup
    rank = init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = DTYPE_MAP.get(dtype, torch.float16)
    is_speculative = draft_checkpoint_path is not None
    
    print(f"Loading model from {checkpoint_path} on {device}")    
    t0 = time.time()
    model = load_model(checkpoint_path, device, precision, use_tp)
    model.requires_grad_(False) 
    multimodal = getattr(model.config, "mm_config", None) is not None
    model._device = device
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
        # Use specified draft device or default to main device
        draft_device = draft_device if draft_device is not None else device
        print(f"Loading draft model from {draft_checkpoint_path} on {draft_device}")
        draft_model = load_model(draft_checkpoint_path, draft_device, precision, use_tp=False)
        draft_model.requires_grad_(False) 
        draft_multimodal = getattr(draft_model.config, "mm_config", None) is not None
        draft_model._device = draft_device
        if draft_multimodal:
            # Set temporary default device for vision modules loading
            if device != draft_device:
                torch.set_default_device(draft_device)
            draft_vision_checkpoints = draft_checkpoint_path.parent / "vision_modules.pth"
            draft_vision_modules = VisionModule.from_name(draft_checkpoint_path.parent.name, 
                                                        config=draft_model.config.mm_config, 
                                                        checkpoint_path=draft_vision_checkpoints,
                                                        dtype=precision,
                                                        device=draft_device)
            draft_vision_modules.eval_mode()
            
            # Restore original default device
            if device != draft_device:
                torch.set_default_device(device)
        else:
            draft_vision_modules = None
    else:
        draft_model = None
        draft_vision_modules = None

    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(checkpoint_path.parent / "tokenizer.model", checkpoint_path)
    
    if compile:
        if is_speculative:
            global model_forward, logits_to_probs, draft_model_forward
            model_forward = torch.compile(model_forward, mode="max-autotune", fullgraph=True)
            draft_model_forward = torch.compile(draft_model_forward, mode="max-autotune", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="max-autotune", fullgraph=True)

        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    questions = load_benchmark_data(bench_name)
    
    conv = get_conversation_template(checkpoint_path)
    
    system_message = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. ")
    
    # Warmup calls
    print(f"Warming up for {warmup} steps...")
    process_questions(
        questions[:warmup], model, tokenizer, conv, system_message,
        max_new_tokens, temperature, top_k, draft_model, speculate_k,
        device, vision_modules, draft_vision_modules,
        max_cache_size, draft_max_cache_size, cross_attention_seq_length, 
        collect_metrics=False, mm_prune_method=mm_prune_method, 
        mm_prune_ratio=mm_prune_ratio
    )

    if num_questions is not None:
        questions = questions[:num_questions]

    eval_info = {
      'model': str(checkpoint_path.parent.name),
      'draft_model': str(draft_checkpoint_path.parent.name) if is_speculative else None,
      'benchmark': bench_name,
      'speculative k': speculate_k if is_speculative else None,
      'compile': compile,
      'compile_prefill': compile_prefill,
      'temperature': temperature,
      'top_k': top_k,
      'use_tp': is_dist_initialized()
    }
    
    print("Running evaluation...")
    results = process_questions(
        questions, model, tokenizer, conv, system_message,
        max_new_tokens, temperature, top_k, draft_model, speculate_k,
        device, vision_modules, draft_vision_modules,
        max_cache_size, draft_max_cache_size, cross_attention_seq_length, 
        collect_metrics=True, eval_info=eval_info,
        mm_prune_method=mm_prune_method, mm_prune_ratio=mm_prune_ratio
    )

    if rank == 0 or rank is None:
        output_file = f"./results_{time.strftime('%Y%m%d-%H%M%S')}.json"
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
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=1, help='Top-k for sampling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation')
    parser.add_argument('--dtype', type=str, default='fp16', help='dtype to use for computation. Choose from fp32, fp16, bf16, int8, int4 and fp8')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Path to the draft model checkpoint for speculative decoding')
    parser.add_argument('--draft_device', type=str, default=None, help='Device to use for draft model (defaults to same as target model)')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill')
    parser.add_argument('--num_questions', type=int, default=None, help='Number of questions to evaluate')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup steps')
    parser.add_argument("--random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--max_cache_size", default=None, type=int, help="Maximum sequence length for the model in cache")
    parser.add_argument("--draft_max_cache_size", default=None, type=int, help="Maximum sequence length for the draft model in cache")
    parser.add_argument("--cross_attention_seq_length", default=None, type=int, help="Maximum cross_attention sequence length for models with cross_attention")
    parser.add_argument('--mm_prune_method', type=str, default=None, 
                       choices=['random', 'structured'],
                       help='Method for pruning multimodal tokens in draft model')
    parser.add_argument('--mm_prune_ratio', type=float, default=0.0,
                       help='Ratio of multimodal tokens to prune (0.0 to 1.0)')
    parser.add_argument('--streaming', action='store_true', help='Enable streaming mode for token-by-token output')
    
    args = parser.parse_args()
    if args.random_seed:
        torch.manual_seed(args.random_seed)
    main(args.bench_name, args.checkpoint_path, args.max_new_tokens, args.temperature, args.top_k, args.device, args.dtype,
         args.draft_checkpoint_path, args.draft_device, args.speculate_k, args.compile, args.compile_prefill, args.num_questions,
         args.warmup, args.max_cache_size, args.draft_max_cache_size, args.cross_attention_seq_length, args.mm_prune_method, 
         args.mm_prune_ratio, args.streaming)