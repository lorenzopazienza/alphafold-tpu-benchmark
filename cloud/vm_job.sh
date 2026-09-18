#!/usr/bin/env bash
# Runs ON the TPU VM, detached (started by tpu_run.sh). Executes a plan, copies
# the results to the bucket, and -- if AUTO_DELETE=1 -- deletes this VM.
set -uo pipefail
PLAN=$1 SESSION=$2 BUCKET=$3 AUTO_DELETE=$4 TPU_NAME=$5 ZONE=$6 PROJECT=$7

cd ~/bench
set -a; . ~/bench/.bench_env; set +a
OUT=~/runs/$SESSION
mkdir -p "$OUT"
cp ~/runs/pip_freeze.txt "$OUT/" 2>/dev/null

~/venv/bin/python cloud/run_with_provenance.py \
  --plan "$PLAN" --out "$OUT" --af2_dir ~/alphafold --src_dir src \
  --python ~/venv/bin/python
echo "runner exit code: $?" | tee -a "$OUT/job.log"

# Results first, deletion second. Only delete if the copy succeeded.
if gcloud storage cp -r "$OUT" "$BUCKET/" ; then
  echo "copied to $BUCKET/$SESSION" | tee -a "$OUT/job.log"
  if [ "$AUTO_DELETE" = "1" ]; then
    gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
  fi
else
  echo "COPY TO BUCKET FAILED -- VM kept; the watchdog will still delete it" | tee -a "$OUT/job.log"
fi
