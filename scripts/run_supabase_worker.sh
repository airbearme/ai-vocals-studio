#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
if [[ -f .env.supabase.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.supabase.local
  set +a
fi

if [[ -z "${SUPABASE_URL:-}" || -z "${SUPABASE_SERVICE_ROLE_KEY:-}" ]]; then
  cat >&2 <<'EOF'
Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY first.

Example:
  export SUPABASE_URL=https://PROJECT_REF.supabase.co
  export SUPABASE_SERVICE_ROLE_KEY=...
  scripts/run_supabase_worker.sh
EOF
  exit 2
fi

exec venv/bin/python supabase_worker.py "$@"
