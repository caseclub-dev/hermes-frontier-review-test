#!/usr/bin/env bash
# Deploy frontier-review-controller.py from repo to runtime location
# Called after git pull on the phase-3c-pr-integration branch
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONTROLLER="$REPO_DIR/scripts/frontier-review-controller.py"
RUNTIME="/home/atlas/.hermes/scripts/frontier-review-controller.py"

if [ ! -f "$CONTROLLER" ]; then
    echo "ERROR: controller not found at $CONTROLLER" >&2
    exit 1
fi

cp "$CONTROLLER" "$RUNTIME"
echo "Deployed: $CONTROLLER -> $RUNTIME"
