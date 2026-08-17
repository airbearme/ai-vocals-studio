#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-airbearme/ai-vocals-studio}"
TARGET_USER="${2:-coden607}"
TARGET_PERMISSION="${3:-push}"
HOST="github.com"
SETTINGS_URL="https://${HOST}/${REPO}/settings/access"

say() { printf '\n%s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

open_url() {
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1" >/dev/null 2>&1 &
  elif command -v gio >/dev/null 2>&1; then
    gio open "$1" >/dev/null 2>&1 &
  else
    say "Open this URL in your browser: $1"
  fi
}

command -v gh >/dev/null 2>&1 || fail "GitHub CLI is required. Install it from https://cli.github.com/"

say "AI Vocals Studio GitHub access setup"
say "Repository: ${REPO}"
say "Target account: ${TARGET_USER}"
say "Requested permission: ${TARGET_PERMISSION}"

if ! gh auth status --hostname "$HOST" >/dev/null 2>&1; then
  say "Step 1 of 4: GitHub login"
  say "A browser window will open. Sign in as the repository owner or an administrator."
  gh auth login --hostname "$HOST" --web --git-protocol https
fi

CURRENT_USER="$(gh api user --hostname "$HOST" --jq .login 2>/dev/null || true)"
[[ -n "$CURRENT_USER" ]] || fail "GitHub authentication was not completed."
say "Authenticated as: ${CURRENT_USER}"

say "Step 2 of 4: Verify repository administration access"
OWNER="$(gh api "repos/${REPO}" --hostname "$HOST" --jq .owner.login 2>/dev/null || true)"
ADMIN="$(gh api "repos/${REPO}" --hostname "$HOST" --jq '.permissions.admin' 2>/dev/null || true)"
[[ "$OWNER" == "airbearme" ]] || fail "This is not the expected airbearme repository."
[[ "$ADMIN" == "true" ]] || fail "${CURRENT_USER} does not have admin access to ${REPO}. Ask the owner or an organization administrator to run this script."

say "Step 3 of 4: Opening repository access settings"
open_url "$SETTINGS_URL"
say "The settings page is open. Review access if needed, then return here."
read -r -p "Press Enter to grant ${TARGET_PERMISSION} access to ${TARGET_USER}, or Ctrl+C to cancel... " _

say "Step 4 of 4: Grant access"
gh api --hostname "$HOST" --method PUT \
  "repos/${REPO}/collaborators/${TARGET_USER}" \
  --field "permission=${TARGET_PERMISSION}" \
  --silent || fail "GitHub rejected the access grant."

say "Access request sent successfully. GitHub may require ${TARGET_USER} to accept an invitation."
say "Verify with: gh api repos/${REPO}/collaborators/${TARGET_USER} --jq .permissions"
say "After access is active, push the production changes with: git push origin main"
