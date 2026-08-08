#!/bin/bash
# Run this on the class server (hpcc-cluster-41) once the TPU quota is confirmed available.
set -euxo pipefail

# --- Step 0: connect to the CORRECT cluster (per Canvas doc, confirmed by Steve Jones) ---
gcloud container clusters get-credentials class-tpu-cluster-west4 --region=us-west4 --project=soe-hpccenter
kubectl config use-context gke_soe-hpccenter_us-west4_class-tpu-cluster-west4
kubectl config current-context   # sanity check: must print the context above

export TEAM=lorenzo

# --- Register our spike script as a ConfigMap ---
kubectl create configmap af-spike-script-${TEAM} --from-file=spike_tpu_forward_pass.py \
  --dry-run=client -o yaml | kubectl apply -f -

# --- Launch the Job ---
envsubst < af_spike_job.yaml | kubectl apply -f -

# --- Watch it ---
kubectl get workloads
kubectl get pods -l job-name=af-spike-${TEAM} -w
