#!/usr/bin/env bash
# Final stage: eval the trained model, build curves + table + model card,
# push to HF Hub, package a local download zip, then (optionally) wipe scratch.
#
# Required:  HF_REPO=your-username/cogito-3b
# Optional:  FINAL=path/to/checkpoint   EVAL_LIMIT=512   CONFIRM_WIPE=1
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"   # activate big-disk venv + cache dirs for eval
: "${HF_REPO:?Set HF_REPO=your-username/cogito-3b}"

FINAL="${FINAL:-}"
if [ -z "$FINAL" ]; then
  for d in checkpoints/math checkpoints/countdown; do
    [ -d "$d/final" ] && FINAL="$d/final" && break
    last="$(ls -d "$d"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
    [ -n "$last" ] && FINAL="$last" && break
  done
fi
[ -z "$FINAL" ] && { echo "[finalize] no checkpoint found; set FINAL=path"; exit 1; }
echo "[finalize] final model: $FINAL"

echo "[finalize] stopping any running training / vLLM server to free all 4 GPUs ..."
bash scripts/stop.sh >/dev/null 2>&1 || true
sleep 5

echo "[finalize] evaluating Cogito (held-out splits) ..."
python eval/eval_countdown.py --model "$FINAL" --tag cogito --limit "${EVAL_LIMIT:-512}" || true
python eval/eval_math.py      --model "$FINAL" --tag cogito --limit "${EVAL_LIMIT:-300}" || true

echo "[finalize] building curves + transcripts + report ..."
python eval/track_length.py || true
python eval/transcripts.py --model "$FINAL" || true
python scripts/make_report.py --model-name "$HF_REPO"
cp results/MODEL_CARD.md "$FINAL/README.md" 2>/dev/null || true

echo "[finalize] pushing to Hugging Face: $HF_REPO"
python scripts/push_to_hub.py --model "$FINAL" --repo "$HF_REPO" --results results

echo "[finalize] packaging local download (cogito_release.zip) ..."
rm -rf release && mkdir -p release/model release/results
cp -r "$FINAL"/. release/model/
cp -r results/. release/results/
( cd release && zip -r ../cogito_release.zip . >/dev/null )
echo "[finalize] wrote $ROOT/cogito_release.zip  (download via the JupyterLab file browser)"

if [ "${CONFIRM_WIPE:-0}" = "1" ]; then
  echo "[finalize] CONFIRM_WIPE=1 -> wiping scratch (checkpoints, datasets, HF cache) ..."
  rm -rf checkpoints data/*.jsonl runs logs
  rm -rf "$ROOT/cache"   # HF + pip + torch + vllm caches all live here (big disk)
  echo "[finalize] wiped. Kept: cogito_release.zip + the HF repo $HF_REPO"
else
  echo "[finalize] Skipped wipe. After you download cogito_release.zip AND confirm"
  echo "[finalize] https://huggingface.co/$HF_REPO looks right, re-run with CONFIRM_WIPE=1."
fi
