# ---- Edit these once ----
export PROJ_DIR=/user/rahulbai/cse676   # where you cloned the repo
export ENV_ACTIVATE="source ~/miniconda3/bin/activate dl676"                    # how you activate your Python env
export SLURM_ACCOUNT=""          # your CCR account/allocation (leave "" to use your default)
# --------------------------
export HF_HOME=$PROJ_DIR/../hf_cache        # keep model caches out of your small home quota
export NLTK_DATA=$PROJ_DIR/../nltk_data
