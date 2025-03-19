#!/bin/bash  

log_dir="logs"
mkdir -p "$log_dir"

export PYTHONUNBUFFERED=1

set -x

export logfile="$log_dir/lamma90B_vision_sd_llama_11B_vision.log"
export checkpoint="checkpoints"
export MODEL_REPO=meta-llama/Llama-3.2-90B-Vision-Instruct
export DRAFT_MODEL_REPO=meta-llama/Llama-3.2-11B-Vision-Instruct
python evaluate.py --checkpoint_path $checkpoint/model/$MODEL_REPO/model.pth \
    --draft_checkpoint_path $checkpoint/model/$DRAFT_MODEL_REPO/model.pth \
    --bench_name MMMU --speculate_k 10 --max_new_tokens 500 --compile --top_k 1 \
    --draft_device "cuda:1" | tee -a "$logfile"