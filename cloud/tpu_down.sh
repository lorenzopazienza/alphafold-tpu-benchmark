#!/usr/bin/env bash
# Delete the TPU VM now and confirm nothing is left running.
set -uo pipefail
: "${PROJECT:?source cloud/env.sh first}"
gcloud compute tpus tpu-vm delete "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --quiet
echo ">> Remaining TPUs in $ZONE (should be empty):"
gcloud compute tpus tpu-vm list --project="$PROJECT" --zone="$ZONE"
