"""Real multi-query batching benchmark for AlphaFold's JAX/Haiku model.

Unlike spike_tpu_forward_pass.py (one protein per call), this uses
jax.vmap to genuinely batch B independent proteins through ONE compiled
XLA call -- true hardware batching, not sequential throughput.

All B proteins in a batch must share the same sequence length (a real
constraint of batched serving: padding/bucketing by length is how real
systems like vLLM handle this too).

Usage:
    python3 spike_batch_forward_pass.py --run_tag=tpu-v5e --batch_size=4 --num_residues=118
"""

import json
import os
import platform
import time

from absl import app
from absl import flags
from absl import logging
import jax

# --- Compatibility shim (same as spike_tpu_forward_pass.py) -----------
import jax.numpy as jnp
_original_clip = jnp.clip
def _clip_compat(*args, **kwargs):
  if "a_min" in kwargs:
    kwargs["min"] = kwargs.pop("a_min")
  if "a_max" in kwargs:
    kwargs["max"] = kwargs.pop("a_max")
  return _original_clip(*args, **kwargs)
jnp.clip = _clip_compat
# ------------------------------------------------------------------------

from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.model import config as af_config
from alphafold.model import features as af_features
from alphafold.model import model as af_model

FLAGS = flags.FLAGS
flags.DEFINE_string("run_tag", None, "Label for this run.")
flags.DEFINE_string("results_dir", "results", "Where to write results.")
flags.DEFINE_integer("num_residues", 118, "Sequence length (shared by every item in the batch).")
flags.DEFINE_integer("num_recycle", 0, "AlphaFold recycle iterations.")
flags.DEFINE_integer("batch_size", 1, "Number of proteins to batch together via jax.vmap.")

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def build_sequence(num_residues, offset=0):
  """Deterministic amino-acid sequence of the given length, offset shifts
  the starting point so different batch items are genuinely different
  sequences (not copies of each other)."""
  alphabet = _AA_ALPHABET[offset % len(_AA_ALPHABET):] + _AA_ALPHABET[:offset % len(_AA_ALPHABET)]
  return (alphabet * (num_residues // len(alphabet) + 1))[:num_residues]


def build_processed_features(sequence, cfg):
  num_res = len(sequence)
  seq_features = pipeline.make_sequence_features(
      sequence=sequence, description="spike_test", num_res=num_res
  )
  toy_msa = parsers.Msa(
      sequences=[sequence],
      deletion_matrix=[[0] * num_res],
      descriptions=["query"],
  )
  msa_features = pipeline.make_msa_features(msas=[toy_msa])
  raw_features = {**seq_features, **msa_features}
  return af_features.np_example_to_features(
      np_example=raw_features, config=cfg, random_seed=0
  )


def main(_):
  backend = jax.default_backend()
  run_tag = FLAGS.run_tag or backend
  num_res = FLAGS.num_residues
  num_recycle = FLAGS.num_recycle
  batch_size = FLAGS.batch_size
  run_id = f"{run_tag}_len{num_res}_recycle{num_recycle}_batch{batch_size}"

  os.makedirs(FLAGS.results_dir, exist_ok=True)
  trace_dir = os.path.join(FLAGS.results_dir, f"trace_{run_id}")

  logging.info("JAX backend: %s", backend)
  logging.info("Run: %s (num_residues=%d, batch_size=%d)", run_id, num_res, batch_size)

  cfg = af_config.model_config("model_3")
  cfg.model.num_recycle = num_recycle
  cfg.data.common.num_recycle = num_recycle
  cfg.data.eval.num_ensemble = 1

  logging.info("Building %d independent %d-residue sequences...", batch_size, num_res)
  per_item_features = [
      build_processed_features(build_sequence(num_res, offset=i), cfg)
      for i in range(batch_size)
  ]

  # Stack the batch_size independent feature dicts along a NEW leading axis.
  # Each item keeps its own internal shape (e.g. the ensemble dim of 1)
  # untouched -- we're adding a genuinely new "which protein" axis on top.
  logging.info("Stacking into a single batched feature dict...")
  stacked_features = jax.tree_util.tree_map(
      lambda *xs: jnp.stack(xs, axis=0), *per_item_features
  )

  logging.info("Constructing RunModel (no trained params -> random init)...")
  runner = af_model.RunModel(cfg, params=None)
  runner.init_params(per_item_features[0], random_seed=0)  # init on one example's shape

  # vmap the model's own apply function over the new batch axis. params and
  # rng are shared (in_axes=None); the feature dict is batched (in_axes=0).
  batched_apply = jax.vmap(runner.apply, in_axes=(None, None, 0))

  logging.info("First batched call -- XLA COMPILE + run (profiler trace on)...")
  t0 = time.time()
  with jax.profiler.trace(trace_dir):
    rng = jax.random.PRNGKey(0)
    result = batched_apply(runner.params, rng, stacked_features)
    jax.block_until_ready(result)
  t_compile_and_run = time.time() - t0
  logging.info("First batched call done in %.1fs (compile + execute)", t_compile_and_run)

  logging.info("Second batched call -- steady-state...")
  t0 = time.time()
  result2 = batched_apply(runner.params, rng, stacked_features)
  jax.block_until_ready(result2)
  t_steady_state = time.time() - t0
  logging.info("Second batched call done in %.3fs (steady-state)", t_steady_state)

  throughput_proteins_per_sec = batch_size / t_steady_state
  per_protein_amortized_s = t_steady_state / batch_size

  memory_stats = []
  for d in jax.devices():
    try:
      stats = d.memory_stats()
      if stats:
        memory_stats.append({
            "device": str(d),
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        })
    except Exception as e:  # pylint: disable=broad-except
      memory_stats.append({"device": str(d), "error": str(e)})

  summary = {
      "run_tag": run_tag,
      "run_id": run_id,
      "backend": backend,
      "num_devices": jax.device_count(),
      "num_residues": num_res,
      "num_recycle": num_recycle,
      "batch_size": batch_size,
      "first_compile_and_run_seconds": round(t_compile_and_run, 2),
      "second_steady_state_seconds": round(t_steady_state, 4),
      "throughput_proteins_per_sec": round(throughput_proteins_per_sec, 3),
      "per_protein_amortized_seconds": round(per_protein_amortized_s, 4),
      "memory_stats_per_device": memory_stats,
  }
  out_path = os.path.join(FLAGS.results_dir, f"result_{run_id}.json")
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

  print("\n" + "=" * 60)
  print("BATCH SPIKE RESULT")
  print("=" * 60)
  for k, v in summary.items():
    if k == "memory_stats_per_device":
      continue
    print(f"  {k:38s}: {v}")
  print(f"  {'results JSON':38s}: {out_path}")
  print("-" * 60)
  for m in memory_stats:
    if "error" in m:
      print(f"    {m['device']}: unavailable")
    else:
      print(f"    {m['device']}: {(m['bytes_in_use'] or 0)/1e6:.0f} MB in use, "
            f"{(m['peak_bytes_in_use'] or 0)/1e6:.0f} MB peak")
  print("=" * 60)


if __name__ == "__main__":
  app.run(main)
