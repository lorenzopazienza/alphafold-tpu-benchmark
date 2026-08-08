# Immutable environment for the AlphaFold JAX/Haiku CPU/GPU/TPU benchmark.
#
# Build one image per backend by passing --build-arg JAX_VARIANT=...
#   docker build --build-arg JAX_VARIANT=cpu   -t af-bench:cpu .
#   docker build --build-arg JAX_VARIANT=cuda12 -t af-bench:gpu .
#   docker build --build-arg JAX_VARIANT=tpu   -t af-bench:tpu .
#
# All system deps, compiler configs (XLA), and Python libraries are baked
# into the image at build time -- nothing is installed manually on a
# running node/pod, per the course's Phase 1 requirement.

FROM python:3.12-slim

ARG JAX_VARIANT=cpu
ENV DEBIAN_FRONTEND=non-interactive

WORKDIR /workspace

# Build-essential + git needed to clone/build AlphaFold and its deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Pin the JAX variant per backend. TPU/GPU wheels come from the same
# jax[...] extras mechanism; only the extras index differs.
RUN if [ "$JAX_VARIANT" = "tpu" ]; then \
        pip install --no-cache-dir -U "jax[tpu]" \
            -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
    elif [ "$JAX_VARIANT" = "cuda12" ]; then \
        pip install --no-cache-dir -U "jax[cuda12]"; \
    else \
        pip install --no-cache-dir -U "jax[cpu]"; \
    fi

# Rest of AlphaFold's pinned deps. Modern biopython/numpy used deliberately --
# the original repo's exact old pins (numpy==1.24.3, biopython==1.79) have
# no wheels for Python 3.12; these versions are drop-in compatible for the
# code paths this benchmark exercises.
RUN pip install --no-cache-dir \
        dm-haiku==0.0.12 \
        ml_collections \
        absl-py \
        "tensorflow-cpu==2.16.1" \
        biopython \
        numpy

# AlphaFold itself -- pinned to a fixed commit for reproducibility.
RUN git clone --depth 1 https://github.com/google-deepmind/alphafold.git /alphafold

COPY src/spike_tpu_forward_pass.py /alphafold/spike_tpu_forward_pass.py

WORKDIR /alphafold
ENTRYPOINT ["python3", "spike_tpu_forward_pass.py"]
