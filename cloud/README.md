# cloud/ — running the benchmark on Cloud TPU (TPU Builders credits)

Tooling for the post-summer measurements on our own Google Cloud project
(`af2-tpu-benchmark`), paid with TPU Builders credits. It does **not** change
`src/`: the same `spike_*.py` scripts run unmodified, exactly as in the summer
GKE jobs (`configs/*.yaml`).

## What it adds over the summer setup

| Problem in the summer data | What this tooling does |
|---|---|
| TPU numbers are single runs | Every configuration runs **n times, each in a fresh process**, interleaved across configurations |
| JAX / host CPU / commit not always recorded (discrepancies 12, 16) | `session.json` records repo commit, AF2 commit, all package versions, Python, CPU model, TPU type, zone, JAX devices, SHA-256 of every script |
| AF2 cloned from `main` without a commit | AF2 pinned to `AF2_COMMIT` in `env.sh` |
| Unpinned Python deps | `requirements-tpu.txt` pins the stack validated on CPU on 2026-09-18 (jax 0.10.2 as in the summer jobs) |
| A forgotten VM keeps billing | Three independent stops: (1) the VM deletes itself when the plan finishes (`AUTO_DELETE=1`), (2) a systemd timer deletes it after `MAX_HOURS` no matter what, (3) `tpu_up.sh` deletes it if setup fails |

Results always go to the bucket **before** any deletion.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `env.sh` | your machine | Project, zone, TPU type, limits. Edit, then `source` |
| `tpu_up.sh` | your machine | Create VM, arm watchdog, copy `src/` + `cloud/`, install stack |
| `setup_vm.sh` | TPU VM | Python 3.12 (uv), pinned deps, AF2 at `AF2_COMMIT`, device check |
| `tpu_run.sh` | your machine | Validate a plan, then start it detached on the VM |
| `vm_job.sh` | TPU VM | Run plan → copy to bucket → self-delete |
| `run_with_provenance.py` | TPU VM (or CPU) | The runner: fresh processes, interleaving, `session.json`, `manifest.jsonl` |
| `tpu_status.sh` | your machine | Progress, and every TPU that exists (= is billing) |
| `tpu_fetch.sh` | your machine | Download a session into `results/cloud/<session>/` |
| `tpu_down.sh` | your machine | Delete now and confirm nothing is left |
| `plans/phase1_v5e8.plan` | — | Repeat of the summer headline measurements, 127 processes |
| `plans/smoke_cpu.plan` | — | Tiny end-to-end check of every script on CPU |

## A session, start to finish

Prerequisites: gcloud installed and logged in (`gcloud auth login`), the
account upgraded from Free Trial, credits applied, Cloud TPU API enabled.

```bash
cd alphafold-tpu-benchmark
source cloud/env.sh
bash cloud/tpu_up.sh                                          # ~10 min
bash cloud/tpu_run.sh cloud/plans/phase1_v5e8.plan 2026-10-02_v5e8
bash cloud/tpu_status.sh 2026-10-02_v5e8                      # any time
# ... VM deletes itself when done ...
bash cloud/tpu_fetch.sh 2026-10-02_v5e8
bash cloud/tpu_status.sh                                      # must list no TPUs
```

Commit `src/` and `cloud/` **before** `tpu_up.sh`: the commit hash is recorded
and `bench_repo_dirty` flags uncommitted changes.

## Output of a session

```
results/cloud/<session>/
  session.json          provenance (see table above)
  manifest.jsonl        one line per process: label, rep, exit code, wall time, files
  pip_freeze.txt        full package list on the VM
  <label>/repNN/        result_*.json written by the spike script + stdout_stderr.log
```

New numbers enter `paper/data/canonical_results.md` as **new rows** citing
`results/cloud/<session>/...`. They do not overwrite the summer rows: a
different JAX build, AF2 commit or host can shift absolute times (see the
Colab August→September level shift).

## CPU smoke test (no cloud needed)

```bash
python3 cloud/run_with_provenance.py --plan cloud/plans/smoke_cpu.plan \
  --out /tmp/smoke --af2_dir /path/to/alphafold --src_dir src \
  --extra_env XLA_FLAGS=--xla_force_host_platform_device_count=8
```

Passed on 2026-09-18 for all five scripts (jax 0.10.2, dm-haiku 0.0.17,
tensorflow-cpu 2.21.0, AF2 `c77e5d2`).

## To verify on first real use

- Runtime names (`v2-alpha-tpuv5-lite`, `v2-alpha-tpuv6e`):
  `gcloud compute tpus versions list --zone=$ZONE`.
- `gcloud` is present on the TPU VM image (the watchdog needs it;
  `tpu_up.sh` aborts and deletes the VM if not).
- The VM's service account may delete TPUs and write to the bucket
  (default Compute Engine account normally has Editor).
- The bucket name is globally unique; change `BUCKET` if taken.
- Quota is not capacity: if creation fails with a stockout, try another zone
  with quota (v5e: us-central1-a/b/c, us-west4-a/c; v6e: us-east5-a/b,
  us-east1-b/d) or set `SPOT=1`.
