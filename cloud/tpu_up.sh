#!/usr/bin/env bash
# Create the TPU VM, arm the self-delete watchdog, copy the repo, install the stack.
# Run from the repo root on a machine with gcloud (your Mac or Cloud Shell):
#   source cloud/env.sh && bash cloud/tpu_up.sh
set -euo pipefail
: "${PROJECT:?source cloud/env.sh first}"

cd "$(git rev-parse --show-toplevel)"
COMMIT=$(git rev-parse HEAD)
DIRTY=$([ -z "$(git status --porcelain -- src cloud)" ] && echo false || echo true)
EXPIRES=$(( $(date +%s) + MAX_HOURS * 3600 ))

SPOT_FLAG=()
[ "$SPOT" = "1" ] && SPOT_FLAG=(--spot)

echo ">> Creating $TPU_NAME ($ACCEL, $RUNTIME) in $ZONE, spot=$SPOT"
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --accelerator-type="$ACCEL" --version="$RUNTIME" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --labels="purpose=af2-bench,owner=lorenzo,expires=$EXPIRES" \
  "${SPOT_FLAG[@]}"

# From here on, if anything fails, delete the VM so it cannot keep billing.
cleanup() {
  echo "!! setup failed -- deleting $TPU_NAME"
  gcloud compute tpus tpu-vm delete "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --quiet || true
}
trap cleanup ERR

SSH=(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --command)

echo ">> Arming watchdog: VM self-deletes in ${MAX_HOURS} h"
"${SSH[@]}" "set -e
  G=\$(command -v gcloud || true)
  if [ -z \"\$G\" ]; then echo 'NO GCLOUD ON VM: watchdog NOT armed' >&2; exit 3; fi
  sudo systemd-run --unit=af2-watchdog --on-active=${MAX_HOURS}h \
    \"\$G\" compute tpus tpu-vm delete $TPU_NAME --project=$PROJECT --zone=$ZONE --quiet
  systemctl list-timers af2-watchdog* --no-pager"

echo ">> Copying src/ and cloud/ (commit $COMMIT, dirty=$DIRTY)"
tar czf /tmp/af2-bench.tgz src cloud
gcloud compute tpus tpu-vm scp /tmp/af2-bench.tgz "$TPU_NAME":~/ --project="$PROJECT" --zone="$ZONE"
"${SSH[@]}" "mkdir -p ~/bench && tar xzf ~/af2-bench.tgz -C ~/bench \
  && printf 'BENCH_COMMIT=%s\nBENCH_DIRTY=%s\n' $COMMIT $DIRTY > ~/bench/.bench_env"

echo ">> Installing the pinned stack on the VM"
"${SSH[@]}" "AF2_COMMIT=$AF2_COMMIT bash ~/bench/cloud/setup_vm.sh"

trap - ERR
echo ">> Ready. Next: bash cloud/tpu_run.sh cloud/plans/<plan> <session_name>"
