#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

MAIN_PY="${MAIN_PY:-venv/bin/python}"
SIDECAR_DIR="${SIDECAR_DIR:-venv_voice_engines}"

if [ ! -x "$MAIN_PY" ]; then
  python3 -m venv venv
  MAIN_PY="venv/bin/python"
fi

echo "Installing Python 3.12-compatible voice engines into main venv..."
"$MAIN_PY" -m pip install --upgrade pip
"$MAIN_PY" -m pip install -r requirements_minimal.txt elevenlabs pyworld demucs

if command -v sox >/dev/null 2>&1; then
  echo "SoX found: $(command -v sox)"
else
  echo "SoX not found. Optional: sudo apt-get install sox libsox-fmt-all ffmpeg"
fi

SIDECAR_PY=""
for candidate in python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    SIDECAR_PY="$candidate"
    break
  fi
done

if [ -z "$SIDECAR_PY" ]; then
  cat <<'EOF'

XTTS/RVC sidecar not installed: Python 3.10 or 3.11 was not found.
Install python3.11 or python3.10, then rerun:
  ./install_voice_engines.sh

The app still has:
  - Qwen3-TTS local neural voice-over
  - ElevenLabs API voice-over when ELEVENLABS_API_KEY is set
  - WORLD/DSP fallback conversion
  - Demucs vocal separation
EOF
  exit 0
fi

echo "Creating optional XTTS/RVC sidecar with $SIDECAR_PY..."
"$SIDECAR_PY" -m venv "$SIDECAR_DIR"
"$SIDECAR_DIR/bin/python" -m pip install --upgrade pip
"$SIDECAR_DIR/bin/python" -m pip install -r requirements_voice_engines_py310.txt

cat > .voice_engine_sidecar <<EOF
VOICE_ENGINE_SIDECAR=${SIDECAR_DIR}/bin/python
EOF

echo "Sidecar installed: ${SIDECAR_DIR}/bin/python"
