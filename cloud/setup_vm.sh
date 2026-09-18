#!/usr/bin/env bash
# Runs ON the TPU VM (called by tpu_up.sh). Installs Python 3.12 via uv, the
# pinned stack, and AlphaFold2 at $AF2_COMMIT, then checks that JAX sees the TPU.
set -euo pipefail
: "${AF2_COMMIT:?}"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 ~/venv
uv pip install --python ~/venv/bin/python -r ~/bench/cloud/requirements-tpu.txt \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

if [ ! -d ~/alphafold ]; then
  git clone https://github.com/google-deepmind/alphafold.git ~/alphafold
fi
git -C ~/alphafold checkout --quiet "$AF2_COMMIT"

mkdir -p ~/runs
uv pip freeze --python ~/venv/bin/python > ~/runs/pip_freeze.txt
~/venv/bin/python -c "import jax; print('backend', jax.default_backend()); print(jax.devices())"
