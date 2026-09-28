#!/usr/bin/env bash
# Publish site/ to GitHub Pages. Copy to scripts/deploy_pages.sh in a project repo.
#
# Why this exists instead of a GitHub Actions workflow: the publishing token in this
# environment is not granted GitHub's `workflow` scope, so any push whose diff touches
# .github/workflows/** is rejected outright, and the Pages REST enablement endpoint is
# unreachable. Pushing a branch named gh-pages whose root is the site auto-enables Pages
# and produces identical output. The workflow that would otherwise do this is parked at
# deploy/github-pages-workflow.yml for anyone whose token can install it.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Fail loudly rather than publishing an empty or half-built site. Point this at a file
# the build step produces — the site's data payload, not just index.html.
REQUIRED="${REQUIRED_ARTIFACT:-site/data/summary.json}"
if [[ ! -f "$REQUIRED" ]]; then
  echo "$REQUIRED is missing — run the report/build step before deploying" >&2
  exit 1
fi

# A uniquely-named build stamp. Verifying a deploy by fetching the site root cannot detect
# a stale publish — the root returns 200 either way. Fetching a path that did not exist
# before this deploy can. Two sibling projects in this series served hours-stale content
# behind a healthy 200 because nobody checked a new path. After this script runs, fetch
# <live-url>/$STAMP and confirm it resolves before calling the deploy done.
STAMP="build-$(date -u +%Y%m%dT%H%M%SZ).txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "site/$STAMP"
echo "$STAMP" > .last_build_stamp

# Publish into docs/ on main as well as gh-pages. Pages can be sourced from either, the
# REST endpoint that would report which is configured is proxy-blocked here, and a repo
# that serves correctly under both configurations cannot be silently mis-served. The cost
# is roughly doubling the site payload in the repo; that trade is deliberate.
rm -rf docs && cp -r site docs && touch docs/.nojekyll

REMOTE="$(git remote get-url origin)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp -r site/. "$TMP/"
# Jekyll would otherwise ignore any path beginning with an underscore.
touch "$TMP/.nojekyll"

cd "$TMP"
git init -q
git checkout -q -b gh-pages
git add -A
git commit -q -m "Publish results site"
git remote add origin "$REMOTE"
git push -q --force origin gh-pages

echo "Pushed gh-pages. Pages will serve it within a minute or two."
echo "Verify with: curl -fsS <live-url>/$STAMP"
