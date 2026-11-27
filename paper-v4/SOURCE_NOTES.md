# Source notes

Some `% src` comments in the LaTeX sources cite evidence that is not a file of this repository: a web page, the abstract of
a cited paper, or the code of AlphaFold2. This file records that evidence. Each section says which sentences of the paper
use it. All pages and code were read on 23 November 2026.

## 1. Cited papers and web pages

| Entry | Sentence of the paper | Evidence |
|---|---|---|
| `googletpuv5e` (Google Cloud, TPU v5e documentation, https://docs.cloud.google.com/tpu/docs/v5e) | II-A: "A TPU v5e chip has 16 GiB of high-bandwidth memory (HBM)." | The page, last updated 2026-11-11, gives "16 GiB" of HBM per chip. Between August and October 2026 the same page said "16 GB". The paper cites the page as read on 2026-11-23. |
| `colabfold2022` (Mirdita et al., Nature Methods, 2022) | I: "ColabFold speeds up batch predictions about 90-fold, in part by avoiding recompilation." | The paper says that ColabFold "speeds up batch predictions by ~90-fold by avoiding recompilation and adding an early stop criterion". |
| `apace2024` (APACE, arXiv:2308.07954 and PNAS, 2024) | I: "APACE runs AlphaFold2 on 200 NVIDIA A100 GPUs." | The abstract, the same on arXiv and in PNAS, says: "Using up to 300 ensembles, distributed across 200 NVIDIA A100 GPUs". "Up to" refers to the ensembles. The GPUs are 200. |

## 2. Shardy guide for JAX users

Entry `shardy2026`: the OpenXLA page "Shardy guide for JAX users", https://openxla.org/shardy/getting_started_jax, read
2026-11-23. The page says that Shardy is now always enabled in JAX. The paper (II-B) reports this and says that we read
the guide on November 23, 2026. Version 4 of the paper also cites the JAX change log (`jaxchangelog`) for the release
in which Shardy became the default partitioner.

## 3. AlphaFold2 at commit c77e5d2

A clone of google-deepmind/alphafold, checked out at commit `c77e5d2a8961d1a353632c462914ff0a32a950f6`
("Update AFS custom template JSON input description", 22 April 2026). On 23 November 2026 this commit was still the head
of the default branch. It tracks 121 files, 87 of them Python files. The September sessions recorded this commit.

**Sharding names (IV-C).** The command

```
git grep -nE 'jax\.sharding|\bMesh\b|PartitionSpec|NamedSharding|with_sharding_constraint|pjit|shard_map|pmap' c77e5d2
```

returns no match in any of the 121 files (exit status 1). The word "sharding" occurs only in a comment of
`alphafold/model/mapping.py` ("shard size None denotes no sharding"), inside `sharded_apply`, a helper that splits a
computation into chunks to save memory.

**Ensemble (II-C).**
- `alphafold/model/modules.py:174-205`: the first ensemble member is computed outside a loop, and the others inside an
  `hk.while_loop`.
- `run_alphafold.py:601-604`: `num_ensemble = 8` for the preset `monomer_casp14`, and `num_ensemble = 1` otherwise.
- Our scripts set `cfg.data.eval.num_ensemble = 1` (`src/spike_tpu_forward_pass.py`, `src/spike_pmap_forward_pass.py`,
  `src/spike_ensemble_shard_forward_pass.py`). With no recycling, one copy of the features remains.

**MSA rows after padding (III-B and IV-E).**
- `alphafold/model/config.py:233`: `max_msa_clusters` is 512. `config.py:144`: `max_extra_msa` is 1024 by default.
- `CONFIG_DIFFS` in `config.py:71-81`: `model_3` and `model_4` set `data.common.max_extra_msa` to 5120. `model_5` changes
  nothing, so it keeps 1024.
- `alphafold/model/tf/data_transforms.py:389-425` (`make_fixed_size`) pads the features to 512 MSA rows and to
  `max_extra_msa` extra MSA rows.
- With one sequence, `msa_feat` has shape [512, L, 49], and `extra_msa` has shape [5120, L] for `model_3` and `model_4`.
  Only the first MSA row holds data. A log of the September runs shows `msa_feat (1, 512, 118, 49)` and
  `extra_msa (1, 5120, 118)`.

**What the seed changes with an MSA of one sequence (IV-C).**
- The seed goes to `tf.set_random_seed`. The external ensemble script passes `random_seed=i` to the featurization.
- With one sequence, the other random steps do nothing: the replacement fraction is zero, the shuffle and the extra MSA
  are empty, and the crop covers the whole sequence.
- The only step that changes is `make_masked_msa` (`alphafold/model/tf/data_transforms.py:354-386`). It selects at random
  about 15% of the positions of the MSA row (`masked_msa_replace_fraction` is 0.15, `config.py:232`) and replaces them.
  This changes `msa_feat` and `bert_mask`. It does not change `aatype` or `target_feat`.

## 4. The 11 single-query runs of the baseline configuration (III-E)

The paper says that the 11 single-query runs of the baseline configuration on the TPU come from seven files. These are
the runs with the single-query script, the fixed sequence of 118 residues, `model_3`, no recycling, and float32
parameters. The key `TPU.single_query_steady_all` of `analysis/out/recomputed.json` lists them.

| File | Runs | Steady-state call (s) |
|---|---|---|
| `results/result_tpu-v5e-podslice.json` | 1 (baseline run) | 0.47 |
| `results/sweep/repeated_runs.json` | 3 | 0.47, 0.469, 0.469 |
| `results/sweep/chip_visibility_sweep.json` | 2 (one and eight chips visible) | 0.469, 0.47 |
| `results/sweep/precision_sweep.json` | 1 (float32) | 0.47 |
| `results/sweep/model_comparison.json` | 1 (`model_3`) | 0.471 |
| `results/sweep/recycle_depth_sweep.json` | 1 (no recycling) | 0.469 |
| `results/sweep/compilation_cache.json` | 2 (first and second process) | 0.47, 0.469 |

In total, these are seven files and 11 runs.
