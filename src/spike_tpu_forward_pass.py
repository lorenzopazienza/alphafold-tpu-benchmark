"""CPU/GPU/TPU backend benchmark for AlphaFold's JAX/Haiku model.

Same script runs unmodified on CPU, GPU, or TPU -- JAX picks up whatever
backend is installed/visible. Supports sweeping sequence length and
recycle depth, so one script covers:
  - the CPU vs GPU vs TPU comparison (Slide 4), and
  - the scale-bounds / profiling study (Slide 3): does the picture change
    as the workload gets bigger, and how many refinement passes (recycles)
    does the real model actually use?

Usage:
    python3 spike_tpu_forward_pass.py --run_tag=tpu-v5e --num_residues=120 --num_recycle=0
    python3 spike_tpu_forward_pass.py --run_tag=cpu --num_residues=250 --num_recycle=1
"""

import json
import os
import platform
import time

from absl import app
from absl import flags
from absl import logging
import jax

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "cache_dir", None,
    "If set, enables JAX's persistent compilation cache at this directory. "
    "A second process pointed at the same directory should reuse the "
    "compiled artifact instead of recompiling -- this is the real "
    "mitigation for the compile-time bottleneck found throughout this "
    "project's other experiments.")
flags.DEFINE_string(
    "model_name", "model_3",
    "Which AlphaFold model config to use. Only model_3/4/5 are supported "
    "here -- model_1/2 default to use_templates=True, which needs "
    "template features this benchmark's minimal pipeline doesn't build.")


# --- Compatibility shim -----------------------------------------------
# AlphaFold's source (a few-years-old repo) calls jnp.clip(x, a_min=..,
# a_max=..) -- current JAX renamed those kwargs to min=/max= and removed
# the old ones. Patch jnp.clip to accept both, so we don't have to hand-edit
# the cloned AlphaFold source. Must happen BEFORE `from alphafold... import`.
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
flags.DEFINE_string(
    "run_tag", None,
    "Label for this run, e.g. 'cpu', 'gpu-t4', 'tpu-v5e'. Defaults to the "
    "detected jax backend if unset.")
flags.DEFINE_string(
    "results_dir", "results",
    "Where to write the per-run JSON result and the profiler trace.")
flags.DEFINE_integer(
    "num_residues", 118,
    "Length of the (synthetic) sequence to fold. Used for the scale-bounds "
    "study -- how does compile/execute time change as this grows?")
flags.DEFINE_integer(
    "num_recycle", 0,
    "Number of AlphaFold recycle iterations (the model's own internal "
    "refinement loop). 0 is fastest; the real model typically uses 3.")
flags.DEFINE_enum(
    "precision", "float32", ["float32", "bfloat16"],
    "Numeric precision for the model's parameters. bfloat16 is one of the "
    "course rubric's own example mitigations for a compute/memory "
    "bottleneck, and TPUs have native bfloat16 matmul support.")

# The 20 standard amino acids, cycled to build a sequence of any requested
# length. Not a real protein for lengths other than 118 (the toy sequence
# below) -- this is a compute/scaling benchmark, not a biology one, so a
# valid-alphabet synthetic sequence is exactly what's needed here.
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

TOY_SEQUENCE_118 = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV"
    "KALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWD"
)


def build_sequence(num_residues):
  """Returns a fixed, reproducible amino-acid sequence of the given length."""
  if num_residues == len(TOY_SEQUENCE_118):
    return TOY_SEQUENCE_118
  return (_AA_ALPHABET * (num_residues // len(_AA_ALPHABET) + 1))[:num_residues]


def build_minimal_features(sequence):
  """Builds a model-ready feature dict without any external search tools."""
  num_res = len(sequence)
  logging.info("Building sequence features for a %d-residue sequence...", num_res)
  seq_features = pipeline.make_sequence_features(
      sequence=sequence, description="spike_test", num_res=num_res
  )
  # Trivial single-sequence MSA: just the query itself, no alignment depth.
  toy_msa = parsers.Msa(
      sequences=[sequence],
      deletion_matrix=[[0] * num_res],
      descriptions=["query"],
  )
  msa_features = pipeline.make_msa_features(msas=[toy_msa])
  return {**seq_features, **msa_features}


def main(_):
  if FLAGS.cache_dir:
    jax.config.update("jax_compilation_cache_dir", FLAGS.cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    logging.info("Persistent compilation cache enabled at: %s", FLAGS.cache_dir)

  backend = jax.default_backend()
  run_tag = FLAGS.run_tag or backend
  num_res = FLAGS.num_residues
  num_recycle = FLAGS.num_recycle
  run_id = f"{run_tag}_{FLAGS.model_name}_len{num_res}_recycle{num_recycle}_{FLAGS.precision}"

  os.makedirs(FLAGS.results_dir, exist_ok=True)
  trace_dir = os.path.join(FLAGS.results_dir, f"trace_{run_id}")

  logging.info("JAX backend: %s", backend)
  logging.info("JAX devices: %s", jax.devices())
  logging.info("Run: %s (num_residues=%d, num_recycle=%d)", run_id, num_res, num_recycle)

  assert FLAGS.model_name in ("model_3", "model_4", "model_5"), (
      "Only model_3/4/5 are supported -- model_1/2 need template features "
      "this benchmark doesn't build.")
  cfg = af_config.model_config(FLAGS.model_name)  # use_templates=False for these three
  cfg.model.num_recycle = num_recycle
  cfg.data.common.num_recycle = num_recycle
  cfg.data.eval.num_ensemble = 1

  sequence = build_sequence(num_res)
  raw_features = build_minimal_features(sequence)

  logging.info("Running TF feature-processing pipeline...")
  processed_features = af_features.np_example_to_features(
      np_example=raw_features, config=cfg, random_seed=0
  )

  logging.info("Constructing RunModel (no trained params -> random init)...")
  runner = af_model.RunModel(cfg, params=None)

  logging.info("Calling init_params (first JIT trace)...")
  t0 = time.time()
  runner.init_params(processed_features, random_seed=0)
  t_init = time.time() - t0
  logging.info("init_params done in %.1fs", t_init)

  if FLAGS.precision == "bfloat16":
    logging.info("Casting params to bfloat16...")
    runner.params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
        runner.params,
    )

  logging.info("First predict() call -- XLA COMPILE + run (profiler trace on)...")
  t0 = time.time()
  with jax.profiler.trace(trace_dir):
    result = runner.predict(processed_features, random_seed=0)
    jax.block_until_ready(result)
  t_compile_and_run = time.time() - t0
  logging.info("First predict() done in %.1fs (compile + execute)", t_compile_and_run)

  logging.info("Second predict() call -- should be compiled already (steady-state)...")
  t0 = time.time()
  result2 = runner.predict(processed_features, random_seed=0)
  jax.block_until_ready(result2)
  t_steady_state = time.time() - t0
  logging.info("Second predict() done in %.2fs (steady-state, no compile)", t_steady_state)

  # Real per-device HBM memory usage via JAX's own API -- works regardless
  # of the container-level metrics setup that tpu-info needed (which wasn't
  # available to us). Not all backends/versions support memory_stats(); we
  # degrade gracefully rather than fail the whole run over a metrics call.
  memory_stats = []
  for d in jax.devices():
    try:
      stats = d.memory_stats()
      if stats:
        memory_stats.append({
            "device": str(d),
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
            "bytes_limit": stats.get("bytes_limit"),
        })
    except Exception as e:  # pylint: disable=broad-except
      memory_stats.append({"device": str(d), "error": str(e)})

  summary = {
      "run_tag": run_tag,
      "run_id": run_id,
      "backend": backend,
      "devices": [str(d) for d in jax.devices()],
      "num_devices": jax.device_count(),
      "host_processor": platform.processor(),
      "host_machine": platform.machine(),
      "num_residues": num_res,
      "num_recycle": num_recycle,
      "model_name": FLAGS.model_name,
      "precision": FLAGS.precision,
      "cache_dir": FLAGS.cache_dir,
      "init_params_seconds": round(t_init, 2),
      "first_predict_compile_and_run_seconds": round(t_compile_and_run, 2),
      "second_predict_steady_state_seconds": round(t_steady_state, 3),
      "memory_stats_per_device": memory_stats,
      "output_final_atom_positions_shape": list(
          result["structure_module"]["final_atom_positions"].shape
      ),
  }
  out_path = os.path.join(FLAGS.results_dir, f"result_{run_id}.json")
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

  print("\n" + "=" * 60)
  print("SPIKE RESULT")
  print("=" * 60)
  for k, v in summary.items():
    if k == "memory_stats_per_device":
      continue  # printed separately below, more readably
    print(f"  {k:38s}: {v}")
  print(f"  {'results JSON':38s}: {out_path}")
  print(f"  {'profiler trace':38s}: {trace_dir}")
  print("-" * 60)
  print("  Per-device HBM memory:")
  for m in memory_stats:
    if "error" in m:
      print(f"    {m['device']}: unavailable ({m['error']})")
    else:
      in_use_mb = (m["bytes_in_use"] or 0) / 1e6
      peak_mb = (m["peak_bytes_in_use"] or 0) / 1e6
      limit_mb = (m["bytes_limit"] or 0) / 1e6
      print(f"    {m['device']}: {in_use_mb:.0f} MB in use, "
            f"{peak_mb:.0f} MB peak, {limit_mb:.0f} MB limit")
  print("=" * 60)


if __name__ == "__main__":
  app.run(main)
