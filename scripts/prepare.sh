# Set default value for download_dir if not provided
DOWNLOAD_DIR=${2:-"checkpoints"}

python scripts/download.py --repo_id $1 --download_dir $DOWNLOAD_DIR && python scripts/convert_hf_checkpoint.py --checkpoint_dir $DOWNLOAD_DIR/$1

# Quantize only if the third parameter (quantization mode) is provided
if [ -n "$3" ]; then
    echo "Quantizing model with mode: $3"
    python quantize.py --checkpoint_path $DOWNLOAD_DIR/$1/model.pth --mode $3
fi