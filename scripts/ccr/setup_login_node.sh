#!/bin/bash
# Run ONCE on the CCR login node (compute nodes may have no internet).
set -e
source "$(dirname "$0")/config.sh"
mkdir -p "$STORAGE_DIR"
[ -d "$ENV_DIR" ] || python3 -m venv "$ENV_DIR"
eval "$ENV_ACTIVATE"
cd "$PROJ_DIR"
mkdir -p slurm_logs "$HF_HOME" "$NLTK_DATA"
chmod -R go-w "$NLTK_DATA"    # NLTK refuses to read data folders other accounts can write to

pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA')"

# Cache keyword/story pairs exactly as train.py / ppo.py / evaluate.py request them
python -c "
from src.utils import load_pairs
for split, n in [('train', 100000), ('validation', 2000), ('train', 20000), ('validation', 4000)]:
    print(split, n, len(load_pairs(split, n)))
"

export HF_HUB_DISABLE_XET=1    # Xet token lookup returns 404 for the legacy gpt2 repo id
# Only config, tokenizer and safetensors files; full repos include weights for every framework.
FILES=(--include "*.json" --include "*.txt" --include "*.model" --include "*.safetensors")
hf download gpt2 "${FILES[@]}"
hf download gpt2-large "${FILES[@]}"    # MAUVE's default feature model
hf download microsoft/Phi-3.5-mini-instruct "${FILES[@]}"    # LLM reward
hf download Qwen/Qwen2.5-7B-Instruct "${FILES[@]}"           # judge
echo "Setup done."
