#!/usr/bin/env bash
# Download a finished session from the bucket into results/cloud/<session>.
set -euo pipefail
: "${PROJECT:?source cloud/env.sh first}"
SESSION=${1:?session name}
cd "$(git rev-parse --show-toplevel)"
mkdir -p results/cloud
gcloud storage cp -r "$BUCKET/$SESSION" results/cloud/
echo ">> results/cloud/$SESSION"
