from pathlib import Path
import json
import ast
from typing import Optional

import torch

from model import Transformer

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
        categories = getattr(bench_args, 'categories',['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
                                                       'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry',
                                                       'Clinical_Medicine', 'Computer_Science', 'Design',
                                                       'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
                                                       'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
                                                       'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
                                                       'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'])
        split = getattr(bench_args, 'split', 'dev')
        questions = []
        for cat in categories:
            dataset =  load_dataset("MMMU/MMMU", cat)[split]
            for i in range(len(dataset)):
                question = {}
                question['cat'] = cat
                turns = dataset[i]['question']
                for element in ast.literal_eval(dataset[i]['options']):
                    if '<image' in element:
                        turns+=element
                if '<image' in dataset[i]['explanation']:
                    turns+=dataset[i]['explanation']
                question['turns'] = [turns]
                question['images'] = []
                question['category'] = cat
                for j in range(1, 8):
                    if dataset[i][f'image_{j}'] is not None:
                        question['images'].append(dataset[i][f'image_{j}'])
                    else:
                        break
                questions.append(question)
    else:
        raise ValueError(f"Unknown benchmark name: {bench_name}")
    return questions


def calculate_sequence_lengths(
    prompt: torch.Tensor,
    max_new_tokens: int,
    embedded: Optional[torch.Tensor],
    draft_encoded: Optional[torch.Tensor],
    draft_embedded: Optional[torch.Tensor],
    speculate_k: Optional[int],
    is_speculative: bool,
    interactive: bool,
    model_block_size: int,
    cross_attention_mask: Optional[torch.Tensor] = None,
    draft_cross_attention_mask: Optional[torch.Tensor] = None,
    max_cache_size: Optional[int] = None,
    draft_max_cache_size: Optional[int] = None,
    cross_attention_seq_length: Optional[int] = None,
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
        draft_embedded: similar to embedded for the draft
        speculate_k: Number of tokens to speculate (if using speculative decoding)
        is_speculative: Whether using speculative decoding
        interactive: Whether in interactive mode
        model_block_size: Maximum sequence length supported by model
        cross_attention_mask: Optional cross attention mask to determine if using cross attention
        draft_cross_attention_mask: Optional draft cross attention mask to determine if using cross attention in draft
        max_cache_size: Optional number for entire sequence length generation to avoid recompilation
        draft_max_cache_size: Optional number for entire draft sequence length generation to avoid recompilation
        cross_attention_seq_length: Optional number for cross attention sequence length to avoid recompilation
    
    Returns:
        Dictionary containing calculated lengths:
        - input_text_length: Length of input text tokens
        - input_embed_length: Length of input embeddings (same as text length for cross-attention case)
        - text_seq_length: Total text sequence length
        - embed_seq_length: Total embedding sequence length
        - draft_text_seq_length: Sequence length for draft model
    """
    input_text_length = prompt.size(-1)
    
    # Determine if we're using cross attention states or direct embeddings
    multimodal = embedded is not None
    is_cross_attention = cross_attention_mask is not None and multimodal
    
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
    if is_speculative:
        draft_input_embed_length = (draft_embedded.size(-2) if (draft_embedded is not None and draft_cross_attention_mask is None) else 
                                    draft_encoded.size(-1) if draft_encoded is not None else 
                                     input_embed_length
        )
        draft_embed_seq_length = draft_input_embed_length + speculate_k + 1 + max_new_tokens 
    else:
        draft_input_embed_length = None
        draft_embed_seq_length = None 
    
    # Verify text-only models have matching lengths
    if not multimodal:
        assert text_seq_length == embed_seq_length, "Text-only model should have the same embed_seq_length as text_seq_length"
        
    return {
        "input_text_length": input_text_length,
        "input_embed_length": input_embed_length,
        "text_seq_length": text_seq_length,
        "embed_seq_length": embed_seq_length,
        "max_cache_size": max_cache_size if max_cache_size is not None else embed_seq_length,
        "draft_input_embed_length": draft_input_embed_length,
        "draft_max_cache_size": draft_max_cache_size if draft_max_cache_size is not None else draft_embed_seq_length,
        "cross_attention_seq_length": cross_attention_seq_length if cross_attention_seq_length is not None else cross_attention_mask.shape[-1] if cross_attention_mask is not None else None,
    }



def load_model(checkpoint_path, device, precision, use_tp):
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

