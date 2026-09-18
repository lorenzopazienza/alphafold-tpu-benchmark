#!/usr/bin/env python3
"""Run benchmark scripts as fresh processes and record full provenance.

Why this exists
---------------
The 2026 summer runs are single measurements, and several of them are missing
the metadata needed to explain later discrepancies (JAX version, host CPU,
repo commit; see canonical_results.md discrepancies 12 and 16). This runner
fixes that for every new Cloud TPU session without touching ``src/``:

* each repeat is a **fresh process**, so cold-start (init + compile) is
  measured n times, not once;
* repeats are **interleaved** across configurations (rep 1 of everything,
  then rep 2, ...), so slow drift of the machine does not masquerade as a
  difference between configurations;
* one ``session.json`` records who/where/what: benchmark repo commit,
  AlphaFold2 commit, package versions, Python, host CPU, TPU type and zone,
  visible JAX devices, and a SHA-256 of every script that was run;
* one line per repeat is appended to ``manifest.jsonl`` (label, repeat,
  exit code, wall time, result JSON paths, log path).

The wrapper itself never imports JAX: importing it would claim the TPU and
block the child processes. Device discovery runs in a short subprocess.

Usage (on the TPU VM, from the benchmark repo root)::

    python3 cloud/run_with_provenance.py \
        --plan cloud/plans/phase1_v5e8.plan \
        --out ~/runs/2026-10-02_v5e8 \
        --af2_dir ~/alphafold

Plan file format: one configuration per line,
``label | script | repeats | extra args``. Blank lines and ``#`` comments are
ignored. ``--results_dir`` is added automatically for each repeat.
"""

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

PACKAGES = [
    "jax", "jaxlib", "libtpu", "libtpu-nightly", "dm-haiku", "tensorflow",
    "tensorflow-cpu", "numpy", "biopython", "ml-collections", "absl-py",
    "chex", "dm-tree",
]

METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/"


def utcnow():
  return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sh(cmd, cwd=None, timeout=60):
  """Runs a command and returns stripped stdout, or None on any failure."""
  try:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                         timeout=timeout, check=True)
    return out.stdout.strip()
  except Exception:  # pylint: disable=broad-except
    return None


def gce_metadata(path):
  """Reads a GCE metadata key; returns None when not on a Google Cloud VM."""
  req = urllib.request.Request(METADATA_URL + path,
                               headers={"Metadata-Flavor": "Google"})
  try:
    with urllib.request.urlopen(req, timeout=2) as resp:
      return resp.read().decode().strip()
  except Exception:  # pylint: disable=broad-except
    return None


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 16), b""):
      h.update(chunk)
  return h.hexdigest()


def cpu_model():
  text = sh(["lscpu"])
  if text:
    for line in text.splitlines():
      if line.lower().startswith("model name"):
        return line.split(":", 1)[1].strip()
  return platform.processor() or None


def mem_total_gb():
  try:
    with open("/proc/meminfo") as f:
      for line in f:
        if line.startswith("MemTotal"):
          return round(int(line.split()[1]) / 1e6, 1)
  except OSError:
    pass
  return None


def jax_devices(python, env):
  """Lists JAX devices in a throwaway subprocess (releases the TPU on exit)."""
  code = ("import jax, json; print(json.dumps({'backend': jax.default_backend(),"
          " 'devices': [str(d) for d in jax.devices()],"
          " 'device_kind': jax.devices()[0].device_kind}))")
  try:
    res = subprocess.run([python, "-c", code], capture_output=True, text=True,
                         timeout=300, env=env, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])
  except Exception as e:  # pylint: disable=broad-except
    return {"error": repr(e)}


def parse_plan(path):
  entries = []
  with open(path) as f:
    for n, raw in enumerate(f, 1):
      line = raw.split("#", 1)[0].strip()
      if not line:
        continue
      parts = [p.strip() for p in line.split("|")]
      if len(parts) != 4:
        sys.exit(f"{path}:{n}: expected 'label | script | repeats | args'")
      label, script, repeats, args = parts
      entries.append({"label": label, "script": script,
                      "repeats": int(repeats), "args": shlex.split(args)})
  labels = [e["label"] for e in entries]
  if len(labels) != len(set(labels)):
    sys.exit(f"{path}: labels must be unique")
  return entries


def schedule(entries, order):
  """Returns (entry, rep) pairs. 'interleaved' = round-robin over repeats."""
  if order == "blocked":
    return [(e, r) for e in entries for r in range(1, e["repeats"] + 1)]
  max_rep = max(e["repeats"] for e in entries)
  return [(e, r) for r in range(1, max_rep + 1) for e in entries
          if r <= e["repeats"]]


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--plan", required=True)
  ap.add_argument("--out", required=True, help="Session output directory.")
  ap.add_argument("--af2_dir", required=True,
                  help="AlphaFold2 checkout (added to PYTHONPATH).")
  ap.add_argument("--src_dir", default="src",
                  help="Directory holding the spike_*.py scripts.")
  ap.add_argument("--python", default=sys.executable)
  ap.add_argument("--order", choices=["interleaved", "blocked"],
                  default="interleaved")
  ap.add_argument("--timeout_min", type=float, default=45,
                  help="Kill a single repeat after this many minutes.")
  ap.add_argument("--dry_run", action="store_true",
                  help="Print the schedule and exit without running anything.")
  ap.add_argument("--extra_env", action="append", default=[],
                  help="KEY=VALUE added to every child's environment "
                       "(e.g. XLA_FLAGS for a CPU smoke test).")
  a = ap.parse_args()

  entries = parse_plan(a.plan)
  for e in entries:
    path = os.path.join(a.src_dir, e["script"])
    if not os.path.isfile(path):
      sys.exit(f"script not found: {path}")
    e["path"] = os.path.abspath(path)
  runs = schedule(entries, a.order)

  if a.dry_run:
    for i, (e, r) in enumerate(runs, 1):
      print(f"{i:4d}  {e['label']:<28} rep {r}  {e['script']} {' '.join(e['args'])}")
    print(f"{len(runs)} processes")
    return

  out = os.path.abspath(os.path.expanduser(a.out))
  os.makedirs(out, exist_ok=True)
  if os.path.exists(os.path.join(out, "manifest.jsonl")):
    sys.exit(f"{out} already has a manifest; use a new --out per session")

  env = dict(os.environ)
  af2 = os.path.abspath(os.path.expanduser(a.af2_dir))
  env["PYTHONPATH"] = af2 + os.pathsep + env.get("PYTHONPATH", "")
  for kv in a.extra_env:
    k, v = kv.split("=", 1)
    env[k] = v

  versions = {}
  for p in PACKAGES:
    try:
      versions[p] = md.version(p)
    except md.PackageNotFoundError:
      pass

  session = {
      "session_start_utc": utcnow(),
      "hostname": socket.gethostname(),
      "command": " ".join(shlex.quote(x) for x in sys.argv),
      "plan_file": os.path.abspath(a.plan),
      "plan_sha256": sha256(a.plan),
      "order": a.order,
      "bench_repo_commit": (os.environ.get("BENCH_COMMIT")
                            or sh(["git", "rev-parse", "HEAD"])),
      "bench_repo_dirty": os.environ.get("BENCH_DIRTY"),
      "af2_dir": af2,
      "af2_commit": sh(["git", "-C", af2, "rev-parse", "HEAD"]),
      "python": sys.version.split()[0],
      "python_executable": a.python,
      "packages": versions,
      "host": {
          "platform": platform.platform(),
          "machine": platform.machine(),
          "cpu_model": cpu_model(),
          "nproc": os.cpu_count(),
          "mem_total_gb": mem_total_gb(),
      },
      "gce": {
          "accelerator_type": gce_metadata("attributes/accelerator-type"),
          "zone": gce_metadata("zone"),
          "machine_type": gce_metadata("machine-type"),
          "preemptible": gce_metadata("scheduling/preemptible"),
          "instance_name": gce_metadata("name"),
      },
      "extra_env": a.extra_env,
      "jax": jax_devices(a.python, env),
      "scripts_sha256": {e["script"]: sha256(e["path"]) for e in entries},
      "plan": [{k: e[k] for k in ("label", "script", "repeats", "args")}
               for e in entries],
  }
  with open(os.path.join(out, "session.json"), "w") as f:
    json.dump(session, f, indent=2)
  print(f"[{utcnow()}] session.json written; {len(runs)} processes scheduled")

  manifest = os.path.join(out, "manifest.jsonl")
  failures = 0
  for i, (e, rep) in enumerate(runs, 1):
    rdir = os.path.join(out, e["label"], f"rep{rep:02d}")
    os.makedirs(rdir, exist_ok=True)
    cmd = [a.python, e["path"], *e["args"], f"--results_dir={rdir}"]
    log_path = os.path.join(rdir, "stdout_stderr.log")
    print(f"[{utcnow()}] {i}/{len(runs)} {e['label']} rep {rep}", flush=True)
    start = utcnow()
    t0 = time.time()
    with open(log_path, "w") as log:
      try:
        rc = subprocess.run(cmd, cwd=rdir, env=env, stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=a.timeout_min * 60).returncode
      except subprocess.TimeoutExpired:
        rc = "timeout"
    wall = round(time.time() - t0, 2)
    results = sorted(os.path.relpath(os.path.join(rdir, fn), out)
                     for fn in os.listdir(rdir)
                     if fn.startswith("result_") and fn.endswith(".json"))
    if rc != 0 or not results:
      failures += 1
    record = {"index": i, "label": e["label"], "rep": rep,
              "script": e["script"], "args": e["args"],
              "start_utc": start, "end_utc": utcnow(),
              "process_wall_seconds": wall, "exit_code": rc,
              "result_files": results,
              "log": os.path.relpath(log_path, out)}
    with open(manifest, "a") as f:
      f.write(json.dumps(record) + "\n")
    print(f"    exit={rc} wall={wall}s results={len(results)}", flush=True)

  session["session_end_utc"] = utcnow()
  session["processes"] = len(runs)
  session["failed_processes"] = failures
  with open(os.path.join(out, "session.json"), "w") as f:
    json.dump(session, f, indent=2)
  print(f"[{utcnow()}] done: {len(runs)} processes, {failures} failed")
  sys.exit(1 if failures else 0)


if __name__ == "__main__":
  main()
