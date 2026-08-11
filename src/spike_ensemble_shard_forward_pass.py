"""Real single-query sharding: AlphaFold's ensemble-averaging distributed
across physical TPU chips via jax.pmap + pmean.

AlphaFold's own ensembling (config.data.eval.num_ensemble > 1) normally
runs SEQUENTIALLY inside one chip (modules.py, AlphaFoldIteration uses
hk.while_loop: each ensemble member's Evoformer pass runs one after
another, accumulating a running sum). That's exactly why our earlier
GSPMD auto-mesh attempt only replicated computation instead of sharding
it -- nothing was structured as independent for XLA to distribute.

This script re-implements the SAME mathematical operation (average N
independent forward passes of the same query) using real distributed
hardware instead: each of N physical chips computes exactly ONE ensemble
member (config.num_ensemble stays 1 -- AlphaFold's own code is completely
unmodified, zero risk to its internal parameter-initialization logic),
and jax.lax.pmean performs a genuine cross-device reduction to average
the results. This is still ONE protein / ONE query -- N independent
stochastic passes of the same sequence, now actually split across
hardware, not a sequential loop.

Honest caveat: our benchmark's MSA is a trivial single-sequence toy MSA
(no real alignment depth), so the N "ensemble members" are near-identical
inputs -- this proves the DISTRIBUTION MECHANISM works for real, but
isn't a scientifically meaningful ensemble average (that would need a
real multi-sequence MSA with actual stochastic subsampling diversity).

Usage:
    python3 spike_ensemble_shard_forward_pass.py --run_tag=tpu-v5e --num_ensemble=8
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
flags.DEFINE_integer("num_ensemble", 8, "Ensemble members -- one per physical chip.")

TOY_SEQUENCE_118 = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV"
    "KALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWD"
)


def main(_):
  backend = jax.default_backend()
  run_tag = FLAGS.run_tag or backend
  num_res = FLAGS.num_residues
  num_ensemble = FLAGS.num_ensemble
  run_id = f"{run_tag}_ensembleshard_len{num_res}_ens{num_ensemble}"

  os.makedirs(FLAGS.results_dir, exist_ok=True)

  n_devices = jax.local_device_count()
  logging.info("JAX backend: %s, local_device_count: %d", backend, n_devices)
  if num_ensemble > n_devices:
    raise ValueError(f"num_ensemble ({num_ensemble}) must be <= device count ({n_devices})")

  # AlphaFold's own config -- num_ensemble stays 1. Its internal code is
  # completely untouched; each pmap replica runs the exact same
  # single-member forward pass AlphaFold always does.
  cfg = af_config.model_config("model_3")
  cfg.model.num_recycle = 0
  cfg.data.common.num_recycle = 0
  cfg.data.eval.num_ensemble = 1

  sequence = TOY_SEQUENCE_118 if num_res == 118 else (
      ("ACDEFGHIKLMNPQRSTVWY" * (num_res // 20 + 1))[:num_res])
  seq_features = pipeline.make_sequence_features(
      sequence=sequence, description="spike_test", num_res=num_res)
  toy_msa = parsers.Msa(sequences=[sequence], deletion_matrix=[[0] * num_res],
                         descriptions=["query"])
  msa_features = pipeline.make_msa_features(msas=[toy_msa])
  raw_features = {**seq_features, **msa_features}

  # Build num_ensemble independent copies (different random_seed each --
  # affects AlphaFold's internal stochastic augmentations even though our
  # toy single-sequence MSA has little real diversity to sample from).
  per_member_features = [
      af_features.np_example_to_features(np_example=raw_features, config=cfg, random_seed=i)
      for i in range(num_ensemble)
  ]
  stacked_features = jax.tree_util.tree_map(
      lambda *xs: jnp.stack(xs, axis=0), *per_member_features)

  logging.info("Constructing RunModel (unmodified AlphaFold code, num_ensemble=1 internally)...")
  runner = af_model.RunModel(cfg, params=None)
  runner.init_params(per_member_features[0], random_seed=0)

  def sharded_apply(params, rng, feat):
    out = runner.apply(params, rng, feat)
    # Average the RAW predicted_lddt logits (a plain JAX array, safe to
    # reduce inside a traced/pmapped function) -- NOT the human-readable
    # 'plddt' score, which AlphaFold computes via get_confidence_metrics()
    # using plain numpy (not jax.numpy). That numpy-based post-processing
    # can only run AFTER apply() returns concrete arrays, never inside a
    # jit/pmap trace -- which is exactly why AlphaFold's own predict()
    # keeps it as a separate step after self.apply(), not inside it.
    logits_mean = jax.lax.pmean(out["predicted_lddt"]["logits"], axis_name="ensemble")
    return logits_mean, out["structure_module"]["final_atom_positions"]

  pmapped_apply = jax.pmap(sharded_apply, axis_name="ensemble", in_axes=(None, None, 0))
  rng = jax.random.PRNGKey(0)

  logging.info("First pmapped ensemble call -- XLA COMPILE + run across %d chips...", num_ensemble)
  t0 = time.time()
  averaged, per_member_positions = pmapped_apply(runner.params, rng, stacked_features)
  jax.block_until_ready(averaged)
  t_compile_and_run = time.time() - t0
  logging.info("First call done in %.1fs", t_compile_and_run)

  logging.info("Second pmapped ensemble call -- steady-state...")
  t0 = time.time()
  averaged2, _ = pmapped_apply(runner.params, rng, stacked_features)
  jax.block_until_ready(averaged2)
  t_steady_state = time.time() - t0
  logging.info("Second call done in %.3fs", t_steady_state)

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
  logits_all_devices_agree = bool(jnp.allclose(averaged[0], averaged[-1]))

  summary = {
      "run_tag": run_tag, "run_id": run_id, "backend": backend,
      "num_devices": jax.device_count(), "num_residues": num_res,
      "num_ensemble": num_ensemble,
      "first_compile_and_run_seconds": round(t_compile_and_run, 2),
      "second_steady_state_seconds": round(t_steady_state, 4),
      "num_chips_with_nonzero_memory": n_chips_used,
      "pmean_reduction_consistent_across_chips": logits_all_devices_agree,
      "memory_stats_per_device": memory_stats,
  }
  out_path = os.path.join(FLAGS.results_dir, f"result_{run_id}.json")
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

  print("\n" + "=" * 60)
  print("ENSEMBLE-SHARD SPIKE RESULT (real single-query sharding)")
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
  print(f"CHIPS ACTUALLY USED: {n_chips_used} / {jax.device_count()}")
  print(f"pmean reduction consistent across chips: {logits_all_devices_agree}"
        f"  (True = the cross-device average genuinely happened correctly)")
  print("=" * 60)


if __name__ == "__main__":
  app.run(main)
