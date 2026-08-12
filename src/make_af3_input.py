"""Builds the AlphaFold3 input JSON for this project's toy sequence.

Shared by the CPU/GPU Colab notebook (notebooks/af3_backend_comparison.ipynb)
and the TPU Kubernetes Job (configs/af_spike_af3_tpu.yaml), so all three
backend runs use the byte-identical input: same 118-residue toy sequence
already used everywhere else in this project (src/spike_tpu_forward_pass.py),
same empty MSA fields, same seed -- the only thing that should change
between runs is --jax_backend.

Usage:
    python3 make_af3_input.py --output_path=af3_toy_input.json
"""

import json

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_path", "af3_toy_input.json",
    "Where to write the AlphaFold3 input JSON.")
flags.DEFINE_integer(
    "seed", 1,
    "Model seed. Kept at 1 to match the existing af3_toy_test_* results "
    "in results/sweep/, so new runs are directly comparable.")

# Same toy sequence as src/spike_tpu_forward_pass.py's TOY_SEQUENCE_118 --
# keep these two definitions in sync if either ever changes.
TOY_SEQUENCE_118 = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV"
    "KALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWD"
)


def build_input(seed):
  """Returns the AlphaFold3 input dict: one chain, no MSA, no templates."""
  return {
      "name": "af3_toy_test",
      "modelSeeds": [seed],
      "sequences": [
          {
              "protein": {
                  "id": "A",
                  "sequence": TOY_SEQUENCE_118,
                  # Deliberately empty -- same systems-first shortcut used
                  # for the AF2 trivial-MSA study: this project is testing
                  # compile/execution behavior, not prediction quality, and
                  # AF3's real genetic-search databases (~252-630GB) are out
                  # of scope. Run with --norun_data_pipeline.
                  "unpairedMsa": "",
                  "pairedMsa": "",
                  "templates": [],
              }
          }
      ],
      "dialect": "alphafold3",
      "version": 1,
  }


def main(argv):
  del argv
  with open(FLAGS.output_path, "w") as f:
    json.dump(build_input(FLAGS.seed), f, indent=2)
  print(f"Wrote {FLAGS.output_path} (seed={FLAGS.seed}, "
        f"{len(TOY_SEQUENCE_118)} residues)")


if __name__ == "__main__":
  app.run(main)
