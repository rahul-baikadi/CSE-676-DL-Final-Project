# ---- Edit these once ----
export PROJ_DIR=/user/rahulbai/cse676/CSE-676-DL-Final-Project   # where you cloned the repo
export STORAGE_DIR=/projects/academic/shaojiet/rahul/cse676       # env and model caches (too big for home quota)
export ENV_DIR=$STORAGE_DIR/dl676-venv                            # created by setup_login_node.sh
export ENV_ACTIVATE="source $ENV_DIR/bin/activate"                # how you activate your Python env
export SLURM_ACCOUNT="shaojiet"        # your CCR account/allocation (leave "" to use your default)
# --------------------------
export HF_HOME=$STORAGE_DIR/hf_cache        # keep model caches out of your small home quota
export NLTK_DATA=$STORAGE_DIR/nltk_data

# CCR's compatibility-layer Python does not search the system library folder, so expose only NVIDIA's libraries.
if compgen -G "/usr/lib/x86_64-linux-gnu/libcuda.so*" > /dev/null; then
  NVIDIA_LIBS=/tmp/$USER-nvidia-libs
  mkdir -p "$NVIDIA_LIBS"
  ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib/x86_64-linux-gnu/libnvidia*.so* "$NVIDIA_LIBS"/ 2>/dev/null || true
  export LD_LIBRARY_PATH=$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
