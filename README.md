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
First install [PyTorch](http://pytorch.org/) according to the instructions specific to your operating system. For AMD GPUs, we strongly recommend using ROCm Software dockers like [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch).

### Download and Convert Model Weights

To download and convert the models listed in the supported model above, use the following command to download the HF model checkpoints:
```cli
bash scripts/prepare.sh <HF_model/repo_id> <download_dir> 
```
where `<HF_model/repo_id>` is the model id from the [HuggingFace](https://huggingface.co/) website. This script will download the model weights from the HuggingFace and then convert them to the format supported by this GPTFast repo. You will need to have your HuggingFace token being added to the environment for the gated models. If you have not done that, you can use this command:
```cli
huggingface-cli login
```


### Run inference



To run vanilla decoding, use the `evaluate.py` script like below:

```cli
python evaluate.py --bench_name MMMU --checkpoint_path   <download_dir>/<HF_model/repo_id>/model.pth`
```

To run speculative decoding, add the draft models arguments as below:

```cli
python evaluate.py --bench_name MMMU --checkpoint_path  <download_dir>/<HF_model_target/repo_id>/model.pth --draft_checkpoint_path  <download_dir>/<HF_model_draft/repo_id>/model.pth --speculate_k <#_of_draft_tokens>`
```
- To compile the model forward passes using `torch.compile()`, you can use the `--compile` flag. Since compilation benefits from a fixed length kv-cache size, it is recommended to use a cache size large enough for both the target and the draft models as below:

```cli
python evaluate.py --bench_name MMMU --checkpoint_path  <download_dir>/<HF_model_target/repo_id>/model.pth  --draft_checkpoint_path <download_dir>/<HF_model_draft/repo_id>/model.pth --speculate_k <#_of_draft_tokens> --compile --max_cache_size <target_model_cache_size> --draft_max_cache_size <target_model_cache_size>
```
- For the Llama 3.2 vision models, it is also preferred to set --cross_attention_seq_length as well to fix the kv-cache size of the cross attention layers.

- To leverage the draft model’s visual token compression for faster speculative decoding, you can use the `--mm_prune_method='random'` or  `--mm_prune_method='structured'` along with `--mm_prune_ratio=<prune_ratio>`.

- For speculative decoding on very large models such as Llama 3.2 90B, you can use the drafter in a seperate gpu with `--draft_device` arguments.


## License

`AMD gpt-fast` is released under the [BSD 3](https://github.com/pytorch-labs/gpt-fast/main/LICENSE) license.

## Acknowledgements

