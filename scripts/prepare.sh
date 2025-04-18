#!/bin/bash

# Set default value for download_dir if not provided
DOWNLOAD_DIR=${2:-"checkpoints"}
REPO_ID=$1

python scripts/download.py --repo_id "$REPO_ID" --download_dir "$DOWNLOAD_DIR"

if [ "$REPO_ID" == "lmms-lab/llava-onevision-qwen2-0.5b-si" ]; then
    rm -f "$DOWNLOAD_DIR/lmms-lab/llava-onevision-qwen2-0.5b-si/training_args.bin"
fi

python scripts/convert_hf_checkpoint.py --checkpoint_dir "$DOWNLOAD_DIR/$REPO_ID"

if [ -n "$3" ]; then
    echo "Quantizing model with mode: $3"
    python quantize.py --checkpoint_path "$DOWNLOAD_DIR/$REPO_ID/model.pth" --mode "$3"
fi