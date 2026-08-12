#!/bin/bash
# CONFIRMED RESULT: this Job fails by design -- AlphaFold3's public
# run_alphafold.py does not support --jax_backend=tpu (valid values are
# cpu|gpu|mps only). Every earlier step succeeds; the flag validation
# itself is the wall, in ~6 seconds. See results/sweep/af3_comparison.md
# Section 3 and results/sweep/af3_tpu_attempt.log for the full writeup
# and log. Kept as documentation of a real attempt, not a working path.
#
# Run this on the class server (hpcc-cluster-41) only to reproduce that
# same confirmed failure.
set -euxo pipefail

gcloud container clusters get-credentials class-tpu-cluster-west4 --region=us-west4 --project=soe-hpccenter
kubectl config use-context gke_soe-hpccenter_us-west4_class-tpu-cluster-west4
kubectl config current-context   # sanity check

export TEAM=lorenzo

# Register the shared input-builder script as a ConfigMap (same file the
# Colab CPU/GPU notebook uses, so all three backends fold the identical input)
kubectl create configmap af3-spike-script-${TEAM} --from-file=../src/make_af3_input.py \
  --dry-run=client -o yaml | kubectl apply -f -

# --- Launch the Job ---
envsubst < ../configs/af_spike_af3_tpu.yaml | kubectl apply -f -

# --- Watch it ---
kubectl get workloads
kubectl get pods -l job-name=af3-spike-tpu-${TEAM} -w

# --- Once the pod shows Completed, copy results out BEFORE the Job/pod is
#     garbage-collected -- results live in an emptyDir, which does not
#     survive pod deletion. ---
echo "Once complete, run:"
echo "  POD=\$(kubectl get pods -l job-name=af3-spike-tpu-${TEAM} -o jsonpath='{.items[0].metadata.name}')"
echo "  kubectl cp \${POD}:/mnt/output/results ./af3_tpu_results"
