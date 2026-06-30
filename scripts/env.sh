#!/usr/bin/env bash
# Redirect EVERY cache + temp dir onto the big disk so /home (small) never fills.
# Source this at the top of every script:  source "<repo>/scripts/env.sh"
# It is safe to source repeatedly. All caches live under $COGITO_HOME/cache, and
# COGITO_HOME defaults to the repo root -- so if you cloned the repo onto the big
# mount (e.g. /lp-dev/<you>/cogito), every cache automatically lands there too.

# Resolve repo root even when this file is *sourced* (BASH_SOURCE, not $0).
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
COGITO_HOME="${COGITO_HOME:-$(cd "$_ENV_DIR/.." && pwd)}"
export COGITO_HOME

_C="$COGITO_HOME/cache"

# Hugging Face (models + datasets) -- the big one (Qwen-3B ~6GB, datasets, etc.)
export HF_HOME="$_C/hf"
export HF_HUB_CACHE="$_C/hf/hub"
export HF_DATASETS_CACHE="$_C/hf/datasets"
export TRANSFORMERS_CACHE="$_C/hf/transformers"   # legacy var; harmless to set

# Torch / Triton / Inductor / vLLM compile caches
export TORCH_HOME="$_C/torch"
export TRITON_CACHE_DIR="$_C/triton"
export TORCHINDUCTOR_CACHE_DIR="$_C/inductor"
export VLLM_CACHE_ROOT="$_C/vllm"

# pip download+build cache, and TMP (flash-attn/vllm builds extract here -> huge)
export PIP_CACHE_DIR="$_C/pip"
export TMPDIR="$_C/tmp"
export TEMP="$_C/tmp"
export TMP="$_C/tmp"

# Generic XDG fallback for anything else that caches to ~/.cache
export XDG_CACHE_HOME="$_C"

# If a virtualenv exists on the big disk (created by setup.sh), activate it so the
# heavy packages (torch/vllm/...) live on the big mount, not the small base disk.
if [ -f "$COGITO_HOME/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$COGITO_HOME/venv/bin/activate"
fi

# Faster HF downloads -- but ONLY if hf_transfer is actually installed, otherwise
# huggingface_hub errors out. Safe to leave off; downloads still work + resume.
if python -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
         "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$VLLM_CACHE_ROOT" "$PIP_CACHE_DIR" "$TMPDIR" 2>/dev/null || true

echo "[env] caches -> $_C  (COGITO_HOME=$COGITO_HOME)"
