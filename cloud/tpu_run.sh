#!/usr/bin/env bash
# Start a plan on the VM, detached (safe to close your laptop).
#   source cloud/env.sh && bash cloud/tpu_run.sh cloud/plans/phase1_v5e8.plan 2026-10-02_v5e8
set -euo pipefail
: "${PROJECT:?source cloud/env.sh first}"
PLAN=${1:?plan file}; SESSION=${2:?session name}

gcloud storage buckets describe "$BUCKET" --project="$PROJECT" >/dev/null 2>&1 \
  || gcloud storage buckets create "$BUCKET" --project="$PROJECT" --location=US

SSH=(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --command)

# Validate the plan on the VM before starting anything long.
"${SSH[@]}" "cd ~/bench && ~/venv/bin/python cloud/run_with_provenance.py --plan $PLAN --out /tmp/dry --af2_dir ~/alphafold --dry_run | tail -1"

# Start detached: the job survives the SSH session ending.
"${SSH[@]}" "cd ~/bench && setsid nohup bash cloud/vm_job.sh $PLAN $SESSION $BUCKET $AUTO_DELETE $TPU_NAME $ZONE $PROJECT \
  > ~/job_$SESSION.log 2>&1 < /dev/null & sleep 5; tail -3 ~/job_$SESSION.log"
echo ">> Running. Check progress with: bash cloud/tpu_status.sh $SESSION"
