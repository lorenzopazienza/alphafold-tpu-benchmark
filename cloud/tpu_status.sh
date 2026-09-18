#!/usr/bin/env bash
# Progress of a running session, plus every TPU that currently exists in the project.
set -uo pipefail
: "${PROJECT:?source cloud/env.sh first}"
SESSION=${1:-}
echo ">> TPUs in $PROJECT (anything listed here is billing):"
gcloud compute tpus tpu-vm list --project="$PROJECT" --zone="$ZONE" \
  --format="table(name,acceleratorType,state,labels.expires)"
if [ -n "$SESSION" ]; then
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" \
    --command "tail -5 ~/job_$SESSION.log; wc -l ~/runs/$SESSION/manifest.jsonl 2>/dev/null" || true
fi
