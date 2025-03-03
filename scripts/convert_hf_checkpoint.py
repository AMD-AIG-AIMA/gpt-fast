# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/llama-2-7b-chat-hf --model_name llama-2.7b-chat
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors_file
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / 'model.safetensors.index.json'
    model_map_json_pytorch = checkpoint_dir / "pytorch_model.bin.index.json"
    model_map_json = None
   
    try:
      assert model_map_json_safetensors.is_file()
      model_map_json = model_map_json_safetensors
      print(f"Found safetensors index at {model_map_json_safetensors}")
    except AssertionError:
      print(f"{model_map_json_safetensors} not found")
    if model_map_json is None:
      try:
        assert model_map_json_pytorch.is_file()
        model_map_json = model_map_json_pytorch
        print(f"Found pytorch index at {model_map_json_pytorch}")
      except AssertionError:
        print(f"{model_map_json_pytorch} not found")
   
    if model_map_json is None:
        print(f"Could not find index file in {checkpoint_dir}")
        model_map_json = create_simple_index_file(checkpoint_dir)

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
        'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.layers.{}.cross_attn.k_norm.weight": "layers.{}.cross_attention.k_norm.weight",
        "model.layers.{}.cross_attn.k_proj.weight": "layers.{}.cross_attention.wk.weight",
        "model.layers.{}.cross_attn.o_proj.weight": "layers.{}.cross_attention.wo.weight",
        "model.layers.{}.cross_attn.q_norm.weight": "layers.{}.cross_attention.q_norm.weight",
        "model.layers.{}.cross_attn.q_proj.weight": "layers.{}.cross_attention.wq.weight",
        "model.layers.{}.cross_attn.v_proj.weight": "layers.{}.cross_attention.wv.weight",
        "model.layers.{}.cross_attn_attn_gate": "layers.{}.cross_attn_attn_gate",
        "model.layers.{}.cross_attn_mlp_gate": "layers.{}.cross_attn_mlp_gate",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        if len(w.shape) == 1:
            # For bias
            return (
                w.unsqueeze(1).view(n_head, 2, config.head_dim // 2, 1)
                .transpose(1, 2)
                .reshape(config.head_dim * n_head, 1).squeeze(1)
            )
        else:
            dim = config.dim
            return (
                w.view(n_head, 2, config.head_dim // 2, dim)
                .transpose(1, 2)
                .reshape(config.head_dim * n_head, dim)
            )

    merged_result = {}
    for file in sorted(bin_files):
       if "safetensors" in str(file):
           state_dict = load_safetensors_file(str(file), device="cpu")
           merged_result.update(state_dict)
       else:
           state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
           merged_result.update(state_dict)
    # Remove 'language_model' prefix from keys
    merged_result = {
        k[len("language_model."):] if k.startswith("language_model.") else k: v 
        for k, v in merged_result.items()
    }
    final_result = {}
    if 'lm_head.weight' not in merged_result:
        merged_result['lm_head.weight'] = merged_result['model.embed_tokens.weight']
    if "llava" in model_name.lower():
        save_llava_vision_parts(merged_result, checkpoint_dir)
    elif "llama" in model_name.lower() and "vision" in model_name.lower():
        save_llama_vision_parts(merged_result, checkpoint_dir)
    elif "qwen2.5" in model_name.lower():
        save_qwen2_5vl_vision_parts(merged_result, checkpoint_dir)
    for key, value in merged_result.items():
        if "llava" in model_name.lower() and key.startswith(
            ("model.vision_tower", "model.mm_projector", "model.image_newline")
        ):
            continue  # Skip these keys for final_result
        elif "llama" in model_name.lower() and key.startswith(
            ("vision_model", "multi_modal_projector")):
            continue  # Skip these keys for final_result
        elif "qwen2.5" in model_name.lower() and key.startswith("visual"):
            continue
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            if 'cross_attention' in key:
                final_result[key] = q
                final_result[key.replace("wq", "wkv")] = torch.cat([k, v])
            else:
                final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
                del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
    if 'llama-3' in model_name.lower():
        # containing 3.1 and 3.2
        if 'llama-3.' in model_name.lower():
            original_dir = checkpoint_dir / "original"
        else:
            original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        tokenizer_model_tiktoken = checkpoint_dir / "tokenizer.model"
        print(f"Copying {tokenizer_model} to {tokenizer_model_tiktoken}")
        shutil.copy(tokenizer_model, tokenizer_model_tiktoken)

def create_simple_index_file(checkpoint_dir: Path):
    from safetensors.torch import safe_open
    # Find all .bin and .safetensors files
    model_files = list(checkpoint_dir.glob('*.bin')) + list(checkpoint_dir.glob('*.safetensors'))
    if not model_files:
        raise FileNotFoundError(f"No .bin or .safetensors files found in {checkpoint_dir}")

    weight_map = {}
    total_size = 0
    for model_file in model_files:
        file_size = model_file.stat().st_size
        total_size += file_size
        if model_file.suffix == '.bin':
            # For .bin files
            state_dict = torch.load(model_file, map_location='cpu')
        else:
            # For .safetensors files
            with safe_open(model_file, framework="pt", device="cpu") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
        
        # Add each weight to the weight_map
        for key in state_dict.keys():
            weight_map[key] = model_file.name
    if model_file.suffix == '.bin':
        index_file = checkpoint_dir / 'pytorch_model.bin.index.json'
    else:
        index_file = checkpoint_dir / 'model.safetensors.index.json'
    # Create the index
    index = {
    "metadata": {"total_size": total_size},
    "weight_map": weight_map
    }
    # Write the index file
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"Created index file: {index_file}")
    return index_file
        
def save_llava_vision_parts(merged_result, checkpoint_dir):
    parts = [
        "vision_tower",
        "mm_projector",
        "image_newline"]
    vision_modules = {}
    for key, value in merged_result.items():
        for part in parts:
            if key.startswith(f"model.{part}"):
                # getting the model. out of the key
                vision_modules[key[6:]] = value
                break
 
    file_path = checkpoint_dir / "vision_modules.pth"
    torch.save(vision_modules, file_path)
    print(f"Saved vision_modules checkpoint to {file_path}")
    
def save_qwen2_5vl_vision_parts(merged_result, checkpoint_dir):
    vision_modules = {}
    for key, value in merged_result.items():
        if key.startswith("visual"):
            # getting the visual. out of the key
            vision_modules[key[7:]] = value
    file_path = checkpoint_dir / "vision_modules.pth"
    torch.save(vision_modules, file_path)
    print(f"Saved vision_modules checkpoint to {file_path}")

def save_llama_vision_parts(merged_result, checkpoint_dir):
    parts = [
        "vision_model",
        "multi_modal_projector"]
    vision_modules = {}
    for key, value in merged_result.items():
        for part in parts:
            if key.startswith(f"{part}"):
                # getting the model. out of the key
                vision_modules[key] = value
                break
 
    file_path = checkpoint_dir / "vision_modules.pth"
    torch.save(vision_modules, file_path)
    print(f"Saved vision_modules checkpoint to {file_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
