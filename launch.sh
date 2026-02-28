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

# Ensure display is set (needed when launched from desktop icon or Claude)
export DISPLAY="${DISPLAY:-:0}"

# Launch — log errors to file so desktop icon failures are debuggable
exec python app_studio.py "$@" 2>>/tmp/ai-vocals-studio.log
