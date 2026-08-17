#!/usr/bin/env bash
set -euo pipefail

# Run this as the airbearme repository owner or an organization administrator.
REPO="${1:-airbearme/ai-vocals-studio}"
USER_TO_GRANT="${2:-coden607}"
PERMISSION="${3:-push}"

case "$PERMISSION" in
  pull|triage|push|maintain|admin) ;;
  *)
    printf 'Permission must be one of: pull, triage, push, maintain, admin\n' >&2
    exit 2
    ;;
esac

command -v gh >/dev/null 2>&1 || {
  printf 'GitHub CLI (gh) is required. Install it from https://cli.github.com/\n' >&2
  exit 1
}

gh auth status >/dev/null 2>&1 || {
  printf 'Authenticate first with: gh auth login\n' >&2
  exit 1
}

printf 'Granting %s permission on %s to %s.\n' "$PERMISSION" "$REPO" "$USER_TO_GRANT"
if [[ "$PERMISSION" == "admin" ]]; then
  printf 'WARNING: admin allows repository settings changes, access management, and deletion.\n'
  read -r -p 'Type GRANT ADMIN to continue: ' confirmation
  [[ "$confirmation" == "GRANT ADMIN" ]] || {
    printf 'Cancelled.\n'
    exit 1
  }
fi

gh api --method PUT \
  "repos/${REPO}/collaborators/${USER_TO_GRANT}" \
  --field "permission=${PERMISSION}" \
  --silent

printf 'Access granted. Verify with:\n'
printf '  gh api repos/%s/collaborators/%s --jq .permissions\n' "$REPO" "$USER_TO_GRANT"
