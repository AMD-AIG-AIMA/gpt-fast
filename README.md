# Multimodal gpt-fast
This is a multimodal version of gpt-fast that adds support for vision-language models, allowing the framework to process both text and images.

Featuring:
1. Very low latency
2. <1000 lines of python
3. No dependencies other than PyTorch and sentencepiece
4. int8/int4 quantization
5. Speculative decoding
6. Tensor parallelism
7. Supports AMD GPUs

This is NOT intended to be a "framework" or "library" - it is intended to show off what kind of performance you can get with native PyTorch :) Please copy-paste and fork as you desire.

For an in-depth walkthrough of what's in this codebase, see this [blog post](link_to_be_added).

## Supported Models

### Text Models
- LLaMA family models (Llama-2, Llama-3, Llama-3.1, Llama-3.2, AMD-Llama)
- Qwen family models (Qwen-2, Qwen-2.5)

### Multimodal Models
This version adds support for several vision-language models:

#### Qwen Vision-Language Models
- Qwen-2.5-VL-3B-Instruct
- Qwen-2.5-VL-7B-Instruct
- Qwen-2.5-VL-72B-Instruct

#### Llava One-Vision Models
- Llava-One-Vision-Qwen2-0.5B-Si
- Llava-One-Vision-Qwen2-7B-Si
- Llava-One-Vision-Qwen2-72B-Si

#### Llama-3.2-Vision-Instruct Models
- Llama-3.2-11B-Vision-Instruct
- Llama-3.2-90B-Vision-Instruct


## Getting Started
### Installation
First install [PyTorch](http://pytorch.org/) according to the instructions specific to your operating system. For AMD GPUs, we strongly recommend to use ROCM dockers like [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch).

### Download and Convert Model Weights

To donwload the models listed in the supported model above use the following command to download the HF model checkpoints:

`python scripts/download.py --repo_id <hf_model_tag> --download_dir <checkpoint_path> --hf_token <your_token>`

To convert the HF model to gpt-fast weight format, use the following command

`python scripts/convert_hf_checkpoint.py --checkpoint_dir <checkpoint_path>/<path_to_model> --model_name <model_name>`

### Run inference

To run vanilla decoding, use the `evaluate.py` script lile below:

`python evaluate.py --bench_name MMMU --checkpoint_path  <checkpoint_path>/<path_to_model>/model.pth`

To run speculative decoding, add the draft models arguments as below:

`python evaluate.py --bench_name MMMU --checkpoint_path  <checkpoint_path>/<path_to_model>/model.pth --draft_checkpoint_path <checkpoint_path>/<path_to_draft_model>/model.pth --speculate_k <#_of_draft_tokens>`

To benefit from `torch.compile()` speed-ups, you can use the `--compile` flag. Since gpt-fast benefits from a fixed length kv-cache size, it is recommended to use a large enough cache size for both target and the draft models as below:

`python evaluate.py --bench_name MMMU --checkpoint_path  <checkpoint_path>/<path_to_model>/model.pth --draft_checkpoint_path <checkpoint_path>/<path_to_draft_model>/model.pth --speculate_k <#_of_draft_tokens> --max_cache_size <target_model_cache_size> --draft_max_cache_size <target_model_cache_size>`

For Llama 3.2 series of models, it is also prefered to set --cross_attention_seq_length as well.

To leverage the draft model’s visual token compression for faster speculative decoding, you can use the `--mm_prune_method='random'` or  `--mm_prune_method='structured'` along with `--mm_prune_ratio=<prune_ratio>`.

For speculative decoding on very large models such as Llama 3.2 90B, you can use the drafter in a seperate gpu with `--draft_device` arguments.


## License

`multimodal gpt-fast` is released under the [BSD 3](https://github.com/pytorch-labs/gpt-fast/main/LICENSE) license.

## Acknowledgements

