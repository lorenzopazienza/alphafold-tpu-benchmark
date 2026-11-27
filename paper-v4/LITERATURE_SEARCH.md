# Literature search on AlphaFold2 inference on TPUs

The introduction of version 4 of the paper says: "On November 23, 2026, we searched OpenAlex, DBLP, arXiv, IEEE
Xplore, Crossref, Europe PMC, the Google Cloud blog, and the GitHub issues of AlphaFold2 and ColabFold. Apart from the
earlier versions of this paper, we found no published measurement of AlphaFold2 inference across a multi-chip TPU
slice." This file records that search.

Version 4 adds four references that a check of the bibliography on November 27, 2026 proposed: the JAX change log,
AlphaFold issues #312 and #611 (already found by the search, section 4), and the 2020 article by Jouppi et al. on the
design of Google's TPUs. This check did not change the result of the search.

The authors ran the search on 23 November 2026, with the help of Claude Code (Anthropic), as the section Use of AI Tools
of the paper states. No query contained the names of the authors, their email addresses, or the arXiv identifier of the
paper. When a query returned the paper itself (arXiv:2609.34818) or its repository, the hit was recorded and not counted
as prior work.

## 1. Question

Is there a published measurement of AlphaFold2, or of a similar protein-structure model, running inference across more
than one chip of a TPU slice or pod?

## 2. Sources

| Group | Sources | Searches | Access |
|---|---|---|---|
| 1 | arXiv (export API), DBLP, Semantic Scholar | 27 planned queries on arXiv, 15 more on arXiv, 42 on DBLP, 27 on Semantic Scholar | arXiv: every result read (title and abstract). DBLP: the search API returned a bot check, so we used the DBLP SPARQL endpoint on titles and read by hand all 187 DBLP titles that contain "AlphaFold". Semantic Scholar: the keyword API answered with HTTP 429 (rate limit), and no result was read. |
| 2 | OpenAlex, Europe PMC, Crossref, IEEE Xplore, web search | about 160 | Google Scholar was not reachable (a session check page). bioRxiv and the ACM Digital Library returned HTTP 403. We used Europe PMC preprints for bioRxiv, Crossref filtered on the ACM member for the ACM Digital Library, and OpenAlex as a broad index. IEEE Xplore was searched directly. |
| 3 | Google Cloud and DeepMind blogs and documentation, GitHub issues of AlphaFold2, ColabFold, and OpenFold, GitHub code and repository search, Kaggle and Colab notebooks | 53 | One page could not be opened (HTTP 999). |
| 4 | Citation chains (Semantic Scholar and OpenAlex) of the works that cite ColabFold, ManyFold, FastFold, ParaFold, APACE, OpenFold, and AlphaFold2, and searches on multi-GPU work | 49 | The citation chains worked. Seven of eight Semantic Scholar keyword searches returned nothing (rate limit). |

The sentence of the paper does not name Semantic Scholar, because its keyword searches returned no results.

## 3. Query terms

Where the source allowed it, every group used these terms: "AlphaFold2 TPU", "AlphaFold TPU inference", "AlphaFold Cloud
TPU", "AlphaFold TPU v4", "AlphaFold TPU v5e", "AlphaFold pmap", "AlphaFold multi-chip", "AlphaFold data parallel
inference", "protein structure prediction TPU", "protein folding TPU benchmark", "ColabFold TPU", "OpenFold TPU",
"ESMFold TPU", "JAX TPU inference benchmark scientific", and "JAX pmap inference throughput TPU". The same terms were
also run with "GPU" in place of "TPU", and only work on the inference performance of more than one device was kept.

## 4. Result

The four groups found, independently, no published measurement of AlphaFold2, ColabFold, OpenFold, or ESMFold inference
across more than one chip of a TPU slice, other than the paper itself.

Work on AlphaFold2 and TPUs that does exist concerns one chip, training, or code without measurements:
- A concurrent working paper (Forsyth III, Zenodo, 14 November 2026, doi:10.5281/zenodo.24224726) measures ColabFold and
  AlphaFold2 inference on one Cloud TPU v5e or v6e chip. It states that it kept one chip per job and leaves multi-chip
  v6e slices to future work. It also covers the persistent compilation cache of JAX. It does not measure `vmap`, `pmap`,
  or partitioning. The paper cites it.
- ColabFold uses the Colab TPU through `jax.tools.colab_tpu`, on one device and without `pmap`. In ColabFold issue #486
  (2023), a user wants to run one job per chip on a TPU VM v2-8, with separate processes, and reports no times.
- ManyFold, which the paper cites, has an AlphaFold forward pass with `vmap` inside `pmap` in its code. Its paper used
  TPUs for training only and timed inference on an A100 GPU.
- A public demonstration on GitHub (ai-infra-demo-proteins) times AlphaFold2 on a TPU v6e and writes "Only 1 chip used".
- DeepMind writes that AlphaFold2 does not support the TPU (AlphaFold issue #611) and that it does not parallelize the
  model over several GPUs (issues #30 and #312). The first official AlphaFold Colab notebook refused TPU runtimes.
- The authors of AlphaFold2 trained the model on TPU v3 and timed inference on V100 GPUs. Inference on the human proteome
  (Tunyasuvunakool et al., Nature, 2021) ran on V100 GPUs. It used several GPUs per worker only for memory.

On GPUs, the works that measure inference on more than one device split one query over the GPUs (FastFold, which the
paper cites, and Fold-CP, arXiv:2603.14806, 2026, for the Boltz models). The works with many independent queries (Summit,
APACE, ParaFold, MassiveFold, and production pipelines) spread them at the level of the workflow. They give no scaling
curve of the network per device. Two works measure many-query inference on CPUs: on the Fugaku supercomputer (Oyama et
al., 2023) and on Intel Xeon processors (Open-Omics-AlphaFold2, 2026).

One review (Zhu, ICEEGT 2025, Atlantis Press) states that AlphaFold inference ran on a TPU v4 slice with a speedup of 11.
The source it cites does not contain this result, so the paper does not cite the review.

## 5. Limits

- Google Scholar could not be checked directly. OpenAlex and a web search restricted to scholar.google.com and
  semanticscholar.org stood in for it.
- The ACM Digital Library was reached only through Crossref, and bioRxiv only through Europe PMC.
- The Semantic Scholar keyword search returned no results. Its citation chains worked.
- A search cannot prove an absence. For this reason the paper says "We found no published measurement" and gives the
  sources and the date.
