#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERCEL_DIR="$ROOT/vercel_frontdoor"
SCOPE="${VERCEL_SCOPE:-stephens-projects-8fbc16d0}"
BUCKET="${SUPABASE_STORAGE_BUCKET:-voiceovers}"

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_vercel_supabase.sh --project-ref <supabase-ref>

Optional env:
  VERCEL_SCOPE=stephens-projects-8fbc16d0
  SUPABASE_STORAGE_BUCKET=voiceovers
  ELEVENLABS_API_KEY=<key>

If ELEVENLABS_API_KEY is not set, the script will use ELEVEN_LABS from .env
when present.
EOF
}

PROJECT_REF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-ref)
      PROJECT_REF="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PROJECT_REF" ]]; then
  usage >&2
  exit 2
fi

command -v supabase >/dev/null || { echo "supabase CLI not found" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq not found" >&2; exit 1; }
command -v npx >/dev/null || { echo "npx not found" >&2; exit 1; }

cd "$ROOT"
echo "[setup] applying Supabase schema to $PROJECT_REF"
supabase db query --linked --project-ref "$PROJECT_REF" --file supabase/schema.sql

SUPABASE_URL="https://${PROJECT_REF}.supabase.co"
echo "[setup] fetching Supabase service role key"
SERVICE_ROLE_KEY="$(
  supabase projects api-keys --project-ref "$PROJECT_REF" --reveal --output json |
  jq -r '.[] | select((.name // .key // .type // "") | test("service"; "i")) | (.api_key // .key // .value)' |
  head -n 1
)"
if [[ -z "$SERVICE_ROLE_KEY" || "$SERVICE_ROLE_KEY" == "null" ]]; then
  echo "Could not find service role key from supabase projects api-keys output." >&2
  exit 1
fi

ELEVEN_KEY="${ELEVENLABS_API_KEY:-}"
if [[ -z "$ELEVEN_KEY" && -f "$ROOT/.env" ]]; then
  ELEVEN_KEY="$(grep -E '^ELEVEN_LABS=' "$ROOT/.env" | head -n 1 | cut -d= -f2- || true)"
fi

add_vercel_env() {
  local key="$1"
  local value="$2"
  local target="${3:-production}"
  if [[ -z "$value" ]]; then
    echo "[setup] skipping empty $key"
    return
  fi
  echo "[setup] setting Vercel env $key ($target)"
  printf '%s\n' "$value" | npx vercel env add "$key" "$target" --scope "$SCOPE" >/tmp/vercel-env-add.log 2>&1 || {
    if grep -qi "already exists" /tmp/vercel-env-add.log; then
      npx vercel env rm "$key" "$target" --yes --scope "$SCOPE" >/dev/null
      printf '%s\n' "$value" | npx vercel env add "$key" "$target" --scope "$SCOPE" >/dev/null
    else
      cat /tmp/vercel-env-add.log >&2
      exit 1
    fi
  }
}

cd "$VERCEL_DIR"
add_vercel_env SUPABASE_URL "$SUPABASE_URL"
add_vercel_env SUPABASE_SERVICE_ROLE_KEY "$SERVICE_ROLE_KEY"
add_vercel_env SUPABASE_STORAGE_BUCKET "$BUCKET"
add_vercel_env ELEVENLABS_API_KEY "$ELEVEN_KEY"

echo "[setup] deploying Vercel production"
npx vercel --prod --yes --scope "$SCOPE"

cat <<EOF

Setup complete.
Supabase URL: $SUPABASE_URL
Vercel: https://ai-vocals-studio.vercel.app

Run local worker:
  scripts/run_supabase_worker.sh
EOF
