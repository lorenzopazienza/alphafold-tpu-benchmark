# Re-analysis of the committed results

These scripts read the result files already in the repository and recompute the numbers of the paper and of version 1
(arXiv:2609.34818v1). They run no experiment and import no code of the project. They read only committed files, and the
August notebooks at commit `faeaa4b` through `git show`.

Run them from the root of the repository:

```bash
python3 -W ignore paper-v4/analysis/recompute.py      # -> paper-v4/analysis/out/recomputed.json and recomputed.md
python3 paper-v4/analysis/check_source_vs_pdf.py      # LaTeX source of version 1 against the PDF of version 1
bash paper-v4/analysis/check_v1_figures.sh /path/to/python-with-matplotlib   # optional
```

| File | What it does | Needs |
|---|---|---|
| `recompute.py` | Recomputes every value that the text uses: baselines, ratios, the September reruns, the profiler teardown from the logs, the TPU repeats, `vmap`, `pmap`, automatic partitioning, the external ensemble, the power-law fit, and the costs. It also lists the input sequence of every experiment with the line of the script, and counts the experiments with result files. `out/recomputed.md` gives the source of every value. | Python 3 standard library, `git` |
| `check_source_vs_pdf.py` | Compares all the numbers of the LaTeX source of version 1 with those of its PDF. | `pdftotext` (poppler) |
| `check_v1_figures.sh` | Copies the scripts `scripts/plot_*.py` to a temporary folder, regenerates the five figures of version 1 from the committed data, and compares their text with the figures of version 1. | `matplotlib`, `pdftotext` |

Notes:
- `recompute.py` uses `datetime.utcfromtimestamp`, which gives a deprecation warning on Python 3.12 and later. The
  warning is harmless, and `python3 -W ignore` hides it.
- The prices of version 1 ($1.20, $0.35, and $0.19 per hour) are inputs taken from `results/cost_analysis.md`, not
  measurements. Versions 3 and 4 use only the TPU price of $1.20 per chip-hour, read on 23 November 2026.
- The 76% of the profiler trace is reported in `profiling/trace_analysis.md`. The raw trace is not in the repository.
