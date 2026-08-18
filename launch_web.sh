#!/bin/bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PORT="${PORT:-8501}"
HOST="${HOST:-0.0.0.0}"
APP_DATA_DIR="${APP_DATA_DIR:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs}"
DATASET_DIR="${DATASET_DIR:-$(pwd)/dataset}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$(pwd)/models}"
PID_FILE="${PID_FILE:-.streamlit/ai-vocals-studio.pid}"
LOG_FILE="${LOG_FILE:-logs/streamlit.log}"

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")" "$OUTPUT_DIR" "$DATASET_DIR" "$MODEL_CACHE_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

venv/bin/python -m pip install -q --upgrade pip
venv/bin/python -m pip install -q -r requirements_minimal.txt

# Best-effort system deps for audio processing (soundfile/sox/ffmpeg support)
if ! command -v sox >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
    sudo -n apt-get install -y -qq sox libsndfile1 ffmpeg >/dev/null 2>&1 || true
fi

if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        kill "$OLD_PID"
        sleep 2
    fi
fi

STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
APP_DATA_DIR="$APP_DATA_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
DATASET_DIR="$DATASET_DIR" \
MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
nohup venv/bin/streamlit run app_streamlit.py \
    --server.port "$PORT" \
    --server.address "$HOST" \
    --server.headless true \
    > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"

for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/_stcore/health" >/dev/null 2>&1; then
        echo "AI Vocals Studio is live: http://localhost:${PORT}"
        exit 0
    fi
    sleep 1
done

echo "Streamlit did not pass health check. See $LOG_FILE" >&2
exit 1
