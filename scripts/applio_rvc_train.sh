#!/usr/bin/env bash
set -euo pipefail

# Adapter used by the queued worker. Applio performs the real four-stage RVC
# pipeline; this script only normalizes its CLI to the worker contract.
DATASET=""
OUTPUT=""
EPOCHS="300"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-300}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

APPLIO_ROOT="${APPLIO_ROOT:-$HOME/Tools/Applio}"
APPLIO_PYTHON="${APPLIO_PYTHON:-$APPLIO_ROOT/.venv/bin/python}"
GPU="${RVC_GPU:--}"
BATCH_SIZE="${RVC_BATCH_SIZE:-4}"
MODEL_NAME="${RVC_MODEL_NAME:-rvc_training_output}"

[[ -d "$DATASET" ]] || { echo "Dataset directory not found: $DATASET" >&2; exit 2; }
[[ -n "$OUTPUT" ]] || { echo "Output directory is required." >&2; exit 2; }
[[ -x "$APPLIO_PYTHON" ]] || {
  echo "Applio is not installed at $APPLIO_ROOT. Run scripts/install_applio_trainer.sh on a GPU worker." >&2
  exit 2
}
mkdir -p "$OUTPUT"

run_applio() {
  (cd "$APPLIO_ROOT" && "$APPLIO_PYTHON" core.py "$@")
}

run_applio preprocess \
  --model_name "$MODEL_NAME" \
  --dataset_path "$DATASET" \
  --sample_rate 40000 \
  --cut_preprocess Automatic \
  --chunk_len 3.0 \
  --overlap_len 0.3 \
  --normalization_mode pre
run_applio extract \
  --model_name "$MODEL_NAME" \
  --sample_rate 40000 \
  --f0_method rmvpe \
  --embedder_model contentvec \
  --gpu "$GPU"
run_applio train \
  --model_name "$MODEL_NAME" \
  --sample_rate 40000 \
  --total_epoch "$EPOCHS" \
  --save_every_epoch 50 \
  --batch_size "$BATCH_SIZE" \
  --gpu "$GPU"
run_applio index --model_name "$MODEL_NAME"

MODEL_DIR="$APPLIO_ROOT/logs/$MODEL_NAME"
MODEL_PATH="$(find "$MODEL_DIR" -type f -name '*.pth' -size +10k -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
INDEX_PATH="$(find "$MODEL_DIR" -type f -name '*.index' -size +0c -print -quit 2>/dev/null || true)"
[[ -n "$MODEL_PATH" ]] || { echo "Applio completed without a valid .pth model." >&2; exit 1; }
cp "$MODEL_PATH" "$OUTPUT/rvc_model.pth"
if [[ -n "$INDEX_PATH" ]]; then cp "$INDEX_PATH" "$OUTPUT/rvc_model.index"; fi
echo "RVC model written to $OUTPUT/rvc_model.pth"
