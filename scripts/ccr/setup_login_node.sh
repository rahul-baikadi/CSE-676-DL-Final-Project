#!/bin/bash
# Run ONCE on the CCR login node (compute nodes may have no internet).
set -e
source "$(dirname "$0")/config.sh"
eval "$ENV_ACTIVATE"
cd "$PROJ_DIR"
mkdir -p slurm_logs "$HF_HOME" "$NLTK_DATA"

pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA')"

# Cache keyword/story pairs exactly as train.py / ppo.py / evaluate.py request them
python -c "
from src.utils import load_pairs
for split, n in [('train', 100000), ('validation', 2000), ('train', 20000), ('validation', 500)]:
    print(split, n, len(load_pairs(split, n)))
"

huggingface-cli download gpt2
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
# Gated: accept the license on huggingface.co first, then `huggingface-cli login`
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct || echo "Llama download failed (gated) - judge will be skipped"
echo "Setup done."