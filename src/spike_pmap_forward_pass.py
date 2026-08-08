"""Real multi-chip data parallelism via jax.pmap.

Unlike spike_batch_forward_pass.py (jax.vmap -- vectorizes WITHIN one chip,
confirmed by that experiment to give zero speedup), this uses jax.pmap to
run B independent proteins on B SEPARATE physical chips simultaneously.

B must be <= the number of visible devices (8 on this slice). Expect wall-
clock time to stay close to the single-protein baseline if this actually
achieves real parallelism, instead of scaling up like the vmap version did.

Usage:
    python3 spike_pmap_forward_pass.py --run_tag=tpu-v5e --batch_size=8 --num_residues=118
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
flags.DEFINE_integer("num_residues", 118, "Sequence length (shared by every item in the batch).")
flags.DEFINE_integer("num_recycle", 0, "AlphaFold recycle iterations.")
flags.DEFINE_integer("batch_size", 8, "Number of proteins, one per physical chip via pmap.")

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def build_sequence(num_residues, offset=0):
  alphabet = _AA_ALPHABET[offset % len(_AA_ALPHABET):] + _AA_ALPHABET[:offset % len(_AA_ALPHABET)]
  return (alphabet * (num_residues // len(alphabet) + 1))[:num_residues]


def build_processed_features(sequence, cfg):
  num_res = len(sequence)
  seq_features = pipeline.make_sequence_features(sequence=sequence, description="spike_test", num_res=num_res)
  toy_msa = parsers.Msa(sequences=[sequence], deletion_matrix=[[0] * num_res], descriptions=["query"])
  msa_features = pipeline.make_msa_features(msas=[toy_msa])
  raw_features = {**seq_features, **msa_features}
  return af_features.np_example_to_features(np_example=raw_features, config=cfg, random_seed=0)


def main(_):
  backend = jax.default_backend()
  run_tag = FLAGS.run_tag or backend
  num_res = FLAGS.num_residues
  num_recycle = FLAGS.num_recycle
  batch_size = FLAGS.batch_size
  run_id = f"{run_tag}_pmap_len{num_res}_recycle{num_recycle}_chips{batch_size}"

  os.makedirs(FLAGS.results_dir, exist_ok=True)

  n_devices = jax.local_device_count()
  logging.info("JAX backend: %s, local_device_count: %d", backend, n_devices)
  if batch_size > n_devices:
    raise ValueError(f"batch_size ({batch_size}) must be <= device count ({n_devices}) for pmap")

  cfg = af_config.model_config("model_3")
  cfg.model.num_recycle = num_recycle
  cfg.data.common.num_recycle = num_recycle
  cfg.data.eval.num_ensemble = 1

  logging.info("Building %d independent %d-residue sequences...", batch_size, num_res)
  per_item_features = [
      build_processed_features(build_sequence(num_res, offset=i), cfg)
      for i in range(batch_size)
  ]
  stacked_features = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *per_item_features)

  logging.info("Constructing RunModel (no trained params -> random init)...")
  runner = af_model.RunModel(cfg, params=None)
  runner.init_params(per_item_features[0], random_seed=0)

  # Replicate params across all chips, and pmap the model's own apply
  # function -- each of the `batch_size` replicas runs on a SEPARATE
  # physical TPU chip, not vectorized within one (that's what vmap did).
  # pmap's in_axes=None automatically broadcasts a shared argument (params,
  # rng) to every device -- no manual replication call needed at all.
  rng = jax.random.PRNGKey(0)
  pmapped_apply = jax.pmap(runner.apply, in_axes=(None, None, 0))

  logging.info("First pmapped call -- XLA COMPILE + run across %d chips...", batch_size)
  t0 = time.time()
  result = pmapped_apply(runner.params, rng, stacked_features)
  jax.block_until_ready(result)
  t_compile_and_run = time.time() - t0
  logging.info("First pmapped call done in %.1fs", t_compile_and_run)

  logging.info("Second pmapped call -- steady-state...")
  t0 = time.time()
  result2 = pmapped_apply(runner.params, rng, stacked_features)
  jax.block_until_ready(result2)
  t_steady_state = time.time() - t0
  logging.info("Second pmapped call done in %.3fs", t_steady_state)

  throughput = batch_size / t_steady_state
  per_protein = t_steady_state / batch_size

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

  summary = {
      "run_tag": run_tag, "run_id": run_id, "backend": backend,
      "num_devices": jax.device_count(), "batch_size": batch_size,
      "num_residues": num_res, "num_recycle": num_recycle,
      "first_compile_and_run_seconds": round(t_compile_and_run, 2),
      "second_steady_state_seconds": round(t_steady_state, 4),
      "throughput_proteins_per_sec": round(throughput, 3),
      "per_protein_amortized_seconds": round(per_protein, 4),
      "num_chips_with_nonzero_memory": n_chips_used,
      "memory_stats_per_device": memory_stats,
  }
  out_path = os.path.join(FLAGS.results_dir, f"result_{run_id}.json")
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

  print("\n" + "=" * 60)
  print("PMAP SPIKE RESULT")
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
  print(f"CHIPS ACTUALLY USED (nonzero memory): {n_chips_used} / {jax.device_count()}")
  print("=" * 60)


if __name__ == "__main__":
  app.run(main)
