#!/bin/bash  

log_dir="logs"
mkdir -p "$log_dir"

export PYTHONUNBUFFERED=1

set -x

export logfile="$log_dir/lamma11B_vision.log"
export checkpoint="checkpoints"
export MODEL_REPO=meta-llama/Llama-3.2-11B-Vision-Instruct
python evaluate.py --checkpoint_path $checkpoint/model/$MODEL_REPO/model.pth \
    --bench_name MMMU --max_new_tokens 500 --compile --top_k 1 | tee "$logfile"