"""EXPLORATORY: single-protein tensor/model sharding via JAX auto-mesh.

Attempts to shard ONE protein's computation across all 8 physical chips
(not multiple proteins across chips like pmap -- one protein's own
internal tensors split across chips). Uses the same GSPMD auto-sharding
pattern the course's own Lab 2 Tunix training script uses:
  mesh = jax.make_mesh(..., axis_types=(jax.sharding.AxisType.Auto,)*N)
  with jax.set_mesh(mesh): ...

This is genuinely exploratory. AlphaFold's code has zero sharding
annotations built in, so XLA's automatic partitioner may or may not find
a beneficial way to split the computation -- we check this empirically via
per-chip memory usage after the run (if sharding worked, HBM should be
spread across multiple chips instead of concentrated on one, unlike every
other experiment in this project).

Usage:
    python3 spike_meshshard_forward_pass.py --run_tag=tpu-v5e --num_residues=118
"""

import json
import os
import time

from absl import app
from absl import flags
from absl import logging
import jax

import jax.numpy as jnp
_original_clip = jnp.clip
def _clip_compat(*args, **kwargs):
  if "a_min" in kwargs:
    kwargs["min"] = kwargs.pop("a_min")
  if "a_max" in kwargs:
    kwargs["max"] = kwargs.pop("a_max")
  return _original_clip(*args, **kwargs)
jnp.clip = _clip_compat

from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.model import config as af_config
from alphafold.model import features as af_features
from alphafold.model import model as af_model

FLAGS = flags.FLAGS
flags.DEFINE_string("run_tag", None, "Label for this run.")
flags.DEFINE_string("results_dir", "results", "Where to write results.")
flags.DEFINE_integer("num_residues", 118, "Sequence length.")
flags.DEFINE_integer("num_recycle", 0, "AlphaFold recycle iterations.")

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
TOY_SEQUENCE_118 = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV"
    "KALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWD"
)


def build_sequence(num_residues):
  if num_residues == len(TOY_SEQUENCE_118):
    return TOY_SEQUENCE_118
  return (_AA_ALPHABET * (num_residues // len(_AA_ALPHABET) + 1))[:num_residues]


def build_minimal_features(sequence):
  num_res = len(sequence)
  seq_features = pipeline.make_sequence_features(sequence=sequence, description="spike_test", num_res=num_res)
  toy_msa = parsers.Msa(sequences=[sequence], deletion_matrix=[[0] * num_res], descriptions=["query"])
  msa_features = pipeline.make_msa_features(msas=[toy_msa])
  return {**seq_features, **msa_features}


def main(_):
  backend = jax.default_backend()
  run_tag = FLAGS.run_tag or backend
  num_res = FLAGS.num_residues
  num_recycle = FLAGS.num_recycle
  run_id = f"{run_tag}_meshshard_len{num_res}_recycle{num_recycle}"
  os.makedirs(FLAGS.results_dir, exist_ok=True)

  logging.info("JAX backend: %s, devices: %d", backend, jax.device_count())

  # Same auto-sharding mesh pattern as the course's own Lab 2 Tunix script.
  n_devices = jax.device_count()
  mesh = jax.make_mesh((n_devices, 1), ("fsdp", "tp"),
                        axis_types=(jax.sharding.AxisType.Auto,) * 2)
  logging.info("Created auto-sharding mesh over %d devices: %s", n_devices, mesh)

  cfg = af_config.model_config("model_3")
  cfg.model.num_recycle = num_recycle
  cfg.data.common.num_recycle = num_recycle
  cfg.data.eval.num_ensemble = 1

  sequence = build_sequence(num_res)
  raw_features = build_minimal_features(sequence)
  processed_features = af_features.np_example_to_features(np_example=raw_features, config=cfg, random_seed=0)

  with jax.set_mesh(mesh):
    logging.info("Constructing RunModel inside mesh context...")
    runner = af_model.RunModel(cfg, params=None)

    logging.info("Calling init_params inside mesh (first JIT trace)...")
    t0 = time.time()
    runner.init_params(processed_features, random_seed=0)
    t_init = time.time() - t0
    logging.info("init_params done in %.1fs", t_init)

    logging.info("First predict() call inside mesh -- XLA COMPILE + run...")
    t0 = time.time()
    result = runner.predict(processed_features, random_seed=0)
    jax.block_until_ready(result)
    t_compile_and_run = time.time() - t0
    logging.info("First predict() done in %.1fs", t_compile_and_run)

    logging.info("Second predict() call -- steady-state...")
    t0 = time.time()
    result2 = runner.predict(processed_features, random_seed=0)
    jax.block_until_ready(result2)
    t_steady_state = time.time() - t0
    logging.info("Second predict() done in %.3fs", t_steady_state)

  memory_stats = []
  for d in jax.devices():
    try:
      stats = d.memory_stats()
      if stats:
        memory_stats.append({"device": str(d), "bytes_in_use": stats.get("bytes_in_use"),
                              "peak_bytes_in_use": stats.get("peak_bytes_in_use")})
    except Exception as e:  # pylint: disable=broad-except
      memory_stats.append({"device": str(d), "error": str(e)})

  n_chips_used = sum(1 for m in memory_stats if m.get("bytes_in_use", 0) and m["bytes_in_use"] > 0)
  # A single (unsharded) chip uses ~463MB for this exact config, per every
  # other experiment in this project. If sharding genuinely split the
  # computation, each chip's memory should be noticeably LOWER than that --
  # not just "nonzero on multiple chips" (which is also what REPLICATION
  # looks like: full-size copies on every chip, not split pieces of one).
  SINGLE_CHIP_BASELINE_MB = 463
  per_chip_mb = [(m.get("bytes_in_use") or 0) / 1e6 for m in memory_stats if "error" not in m]
  avg_chip_mb = sum(per_chip_mb) / len(per_chip_mb) if per_chip_mb else 0
  sharding_worked = n_chips_used > 1 and avg_chip_mb < SINGLE_CHIP_BASELINE_MB * 0.8

  summary = {
      "run_tag": run_tag, "run_id": run_id, "backend": backend,
      "num_devices": jax.device_count(), "num_residues": num_res, "num_recycle": num_recycle,
      "init_params_seconds": round(t_init, 2),
      "first_predict_compile_and_run_seconds": round(t_compile_and_run, 2),
      "second_predict_steady_state_seconds": round(t_steady_state, 3),
      "num_chips_with_nonzero_memory": n_chips_used,
      "sharding_appears_to_have_worked": sharding_worked,
      "avg_hbm_per_chip_mb": round(avg_chip_mb, 1),
      "single_chip_baseline_mb": SINGLE_CHIP_BASELINE_MB,
      "memory_stats_per_device": memory_stats,
  }
  out_path = os.path.join(FLAGS.results_dir, f"result_{run_id}.json")
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

  print("\n" + "=" * 60)
  print("MESH-SHARD SPIKE RESULT (exploratory)")
  print("=" * 60)
  for k, v in summary.items():
    if k == "memory_stats_per_device":
      continue
    print(f"  {k:38s}: {v}")
  print("-" * 60)
  for m in memory_stats:
    if "error" in m:
      print(f"    {m['device']}: unavailable")
    else:
      print(f"    {m['device']}: {(m['bytes_in_use'] or 0)/1e6:.0f} MB in use, "
            f"{(m['peak_bytes_in_use'] or 0)/1e6:.0f} MB peak")
  print("=" * 60)
  print(f"CHIPS WITH NONZERO MEMORY: {n_chips_used} / {jax.device_count()}"
        f" -- {'sharding likely worked' if sharding_worked else 'XLA replicated the full computation instead of sharding it'} "
        f"(avg {avg_chip_mb:.0f}MB/chip vs {SINGLE_CHIP_BASELINE_MB}MB single-chip baseline)")
  print("=" * 60)


if __name__ == "__main__":
  app.run(main)
