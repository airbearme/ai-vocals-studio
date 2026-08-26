#!/usr/bin/env bash
set -euo pipefail

ROOT="${APPLIO_ROOT:-$HOME/Tools/Applio}"
MIN_FREE_KB=$((10 * 1024 * 1024))
FREE_KB="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"

if [[ "$FREE_KB" -lt "$MIN_FREE_KB" ]]; then
  echo "At least 10 GB free disk space is required for Applio training." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA GPU detected. Install Applio on a GPU worker or cloud GPU; CPU training is impractical for this application." >&2
  exit 2
fi
command -v git >/dev/null || { echo "git is required." >&2; exit 2; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 2; }

if [[ ! -d "$ROOT/.git" ]]; then
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 https://github.com/IAHispano/Applio.git "$ROOT"
fi
cd "$ROOT"
chmod +x run-install.sh run-applio.sh
./run-install.sh
echo "Applio installed. Set RVC_TRAINER=$OLDPWD/scripts/applio_rvc_train.sh before starting the worker."
