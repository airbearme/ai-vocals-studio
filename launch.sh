#!/bin/bash
# AI Vocals Studio Pro — one-click launcher
cd "$(dirname "$(readlink -f "$0")")"

# Create venv if missing
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install / update deps silently
pip install -q gtts librosa numpy soundfile pydub 2>/dev/null

# Launch
exec python app_studio.py "$@"
