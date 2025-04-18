# Multimodal gpt-fast

![Demo](./media/MMSpecDec.gif)

This is a multimodal version of GPT-Fast that adds support for vision-language models, allowing the framework to process both text and images.

Featuring:
1. Very low latency
2. <1000 lines of Python
3. No dependencies other than PyTorch and Transformers
4. int8/int4/fp8 quantizations
5. Speculative decoding
6. Tensor parallelism
7. Supports AMD GPUs

This is NOT intended to be a "framework" or "library" - it is intended to show off what kind of performance you can get with native PyTorch :) Please copy-paste and fork as you desire.

For an in-depth walkthrough of what's in this codebase, see this [blog post](link_to_be_added).

## Supported Models

### Text Models
- LLaMA family models (Llama-2, Llama-3, Llama-3.1, Llama-3.2, AMD-Llama) (Example 🤗: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main), [amd/AMD-Llama-135m](https://huggingface.co/amd/AMD-Llama-135m), ...)

- Qwen family models (Qwen-2, Qwen-2.5) (Example 🤗: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), ...)


### Multimodal Models
This version adds support for several vision-language models:

#### Qwen Vision-Language Models
- Qwen-2.5-VL-3B-Instruct (🤗 [Qwen/Qwen2.5-VL-3B-Instruct](Qwen/Qwen2.5-VL-3B-Instruct))

- Qwen-2.5-VL-7B-Instruct (🤗 [Qwen/Qwen2.5-VL-7B-Instruct](Qwen/Qwen2.5-VL-7B-Instruct))

- Qwen-2.5-VL-72B-Instruct (🤗 [Qwen/Qwen2.5-VL-72B-Instruct](Qwen/Qwen2.5-VL-3B-Instruct))


#### Llava One-Vision Models
- Llava-One-Vision-Qwen2-0.5B-Si (🤗 [lmms-lab/llava-onevision-qwen2-0.5b-si](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-si))

- Llava-One-Vision-Qwen2-7B-Si (🤗 [lmms-lab/llava-onevision-qwen2-7b-si](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-si))

- Llava-One-Vision-Qwen2-72B-Si (🤗 [lmms-lab/llava-onevision-qwen2-72b-si](https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-si))


#### Llama-3.2-Vision-Instruct Models
- Llama-3.2-11B-Vision-Instruct (🤗 [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct))

- Llama-3.2-90B-Vision-Instruct (🤗 [meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct))



## Getting Started
### Installation
First, install [PyTorch](http://pytorch.org/) according to the instructions specific to your operating system. For AMD GPUs, we strongly recommend using ROCm Software dockers like [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch).
You can install the required packages using the command below to avoid reinstalling Torch from scratch.
```bash
pip install -r requirements.txt -c constraints.txt
```

### Download and Convert Model Weights

To download and convert the models listed in the supported model above, use the following command to download the HF model checkpoints:
```bash
bash scripts/prepare.sh <HF_model/repo_id> <download_dir> 
```
where `<HF_model/repo_id>` is the model id from the [HuggingFace](https://huggingface.co/) website. This script will download the model weights from HuggingFace and then convert them to the format supported by this GPTFast repo. You will need to have your HuggingFace token added to the environment for the gated models. If you have not done that, you can use this command:
```bash
huggingface-cli login
```
### Optional: Quantize Model Weights
To save memory and potentially improve performance, you can quantize models to int8, int4, or fp8:

```bash
python quantize.py --checkpoint_path <download_dir>/<HF_model/repo_id>/model.pth --mode int8
```
You can also directly apply quantization when preparing models by adding the quantization mode as a third parameter:
```bash
bash scripts/prepare.sh <HF_model/repo_id> <download_dir> int8
```

### Run inference

#### Benchmarking
To run vanilla decoding benchmarks, use the `evaluate.py` script like below:

```bash
python evaluate.py --bench_name MMMU --checkpoint_path   <download_dir>/<HF_model/repo_id>/model.pth`
```

To run speculative decoding, add the draft models' arguments as below:

```bash
python evaluate.py --bench_name MMMU --checkpoint_path  <download_dir>/<HF_model_target/repo_id>/model.pth --draft_checkpoint_path  <download_dir>/<HF_model_draft/repo_id>/model.pth --speculate_k <\#_of_draft_tokens>`
```
- To compile the model forward passes using `torch.compile()`, you can use the `--compile` flag. Since compilation benefits from a fixed length kv-cache size, it is recommended to use a cache size large enough for both the target and the draft models as below by setting the `--max_cache_size` and `--draft_max_cache_size` arguments:

```bash
python evaluate.py --bench_name MMMU --checkpoint_path  <download_dir>/<HF_model_target/repo_id>/model.pth  --draft_checkpoint_path <download_dir>/<HF_model_draft/repo_id>/model.pth --speculate_k <\#_of_draft_tokens> --compile --max_cache_size <target_model_cache_size> --draft_max_cache_size <target_model_cache_size>
```
- For the Llama 3.2 vision models, it is also preferred to set `--cross_attention_seq_length` as well to fix the kv-cache size of the cross attention layers.

- To leverage the draft model’s visual token compression for faster speculative decoding, you can use the `--mm_prune_method='random'` or  `--mm_prune_method='structured'` along with `--mm_prune_ratio=<prune_ratio>`.

- For speculative decoding on very large models such as Llama 3.2 90B, you can use the drafter in a seperate gpu with `--draft_device` arguments.

- To use Tensor Parallel distributed strategy for large multimodal models, you can prepend `ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=<#_gpus>` to the above commands.

#### Interactive Text Generation with Web UI
To run the Gradio app to interact with the model, use the following command. If you have not installed the Gradio library, you can install it using the command below:

```bash
pip install gradio
```

Now you can run the app with the following command:
```bash
python app.py --checkpoint_path <download_dir>/<HF_model/repo_id>/model.pth
```

To use speculative decoding, add the following arguments:

```bash
python app.py --checkpoint_path <download_dir>/<HF_model/repo_id>/model.pth --speculate_k <#_of_draft_tokens>
```

The web UI automatically detects if your model is multimodal and displays an image upload interface if it is. You can:
- Upload images
- Adjust temperature and other sampling parameters
- Toggle speculative decoding on/off
- Stream generated text in real-time

## License

`AMD Multimodal gpt-fast` is released under the same license as the original GPTFast, [BSD 3](https://github.com/pytorch-labs/gpt-fast/main/LICENSE) license.

## Acknowledgements
This project builds upon the original GPT-Fast by the PyTorch team and extends it with multimodal capabilities.
