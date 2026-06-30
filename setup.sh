#!/usr/bin/env bash
# One-shot environment setup on the GPU box. Idempotent-ish; safe to re-run.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Route ALL caches/temp onto the big disk BEFORE any pip/HF activity.
source "$ROOT/scripts/env.sh"

echo "[setup] python: $(python --version 2>&1)"
echo "[setup] repo + caches on: $ROOT"
echo "[setup] free space here:"
df -h "$ROOT" 2>/dev/null || true

# Create a virtualenv ON THE BIG DISK so torch/vllm/... (~10GB) never touch the
# small base disk. env.sh auto-activates it for the launch/serve scripts later.
if [ ! -f "$ROOT/venv/bin/activate" ]; then
  echo "[setup] creating virtualenv at $ROOT/venv (on the big disk)"
  python -m venv "$ROOT/venv"
fi
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
echo "[setup] using python: $(command -v python)"

# tmux + curl (best-effort; training falls back to nohup if tmux is absent)
if ! command -v tmux >/dev/null 2>&1; then
  (apt-get update -y && apt-get install -y tmux curl) >/dev/null 2>&1 || \
  echo "[setup] could not apt-get tmux (non-root?). Will use nohup fallback."
fi

echo "[setup] upgrading pip toolchain"
pip install -U pip setuptools wheel

echo "[setup] installing RL stack (vllm, trl, transformers, ...)"
pip install -r "$ROOT/requirements.txt"

echo "[setup] installing hf_transfer (faster/resumable HF downloads)"
pip install hf_transfer || echo "[setup] hf_transfer optional install failed; continuing"

echo "[setup] installing flash-attn (optional; falls back to sdpa if it fails)"
pip install flash-attn --no-build-isolation || echo "[setup] flash-attn not installed -> sdpa attention"

echo "[setup] GPU sanity check"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("  GPU %d: %s (%.0f GB)" % (i, p.name, p.total_memory / 1e9))
PY
nvidia-smi || true
echo "[setup] DONE. Next: huggingface-cli login  ->  python data/build_countdown.py"
