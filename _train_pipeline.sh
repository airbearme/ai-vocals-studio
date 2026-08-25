#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "venv/bin/activate"
fi

SPEAKER="${1:-Pacaveli}"
SOURCE_DIR="${SOURCE_DIR:-dataset/${SPEAKER}_processed}"
RAW_DIR="${RAW_DIR:-dataset_raw/${SPEAKER}}"

mkdir -p "$RAW_DIR" "dataset/44k/${SPEAKER}" "configs/44k" "logs/44k"

shopt -s nullglob
sources=("$SOURCE_DIR"/*.wav)
if (( ${#sources[@]} == 0 )); then
  echo "[error] No WAV files found in $SOURCE_DIR" >&2
  exit 1
fi

cp "${sources[@]}" "$RAW_DIR/"

echo "=== Files in $RAW_DIR ==="
ls -la "$RAW_DIR"

if ! command -v svc >/dev/null 2>&1; then
  echo "[error] svc command not found. Install so-vits-svc-fork in the active environment." >&2
  exit 1
fi

echo "=== Running pre-resample ==="
svc pre-resample 2>&1 | tail -40

echo "=== Done with preprocessing check ==="
