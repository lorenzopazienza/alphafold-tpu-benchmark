# Configuration for one Cloud TPU session. Edit, then `source cloud/env.sh`.
# Every script in cloud/ reads these variables.

export PROJECT="af2-tpu-benchmark"
export ZONE="us-central1-a"            # v5e quota 16 here; v6e: us-east5-a or us-east1-d
export TPU_NAME="af2-v5e-8"
export ACCEL="v5litepod-8"             # v6e: v6e-8
export RUNTIME="v2-alpha-tpuv5-lite"   # v6e: v2-alpha-tpuv6e  (check: gcloud compute tpus versions list --zone=$ZONE)
export SPOT=0                          # 1 = Spot VM (cheaper, can be preempted; fine for repeats)
export MAX_HOURS=6                     # hard limit: the VM deletes itself after this many hours
export AUTO_DELETE=1                   # 1 = the VM deletes itself as soon as the plan finishes
export BUCKET="gs://af2-tpu-benchmark-results"   # results are copied here before any deletion

# AlphaFold2 commit used for every Cloud session (validated on CPU 2026-09-18).
# The summer runs cloned main without recording the commit.
export AF2_COMMIT="c77e5d2a8961d1a353632c462914ff0a32a950f6"
