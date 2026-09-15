# Website and Slide Updates Needed

Numbers and claims on the public website, the slide deck and the READMEs that no
longer match the corrected data in `paper/data/canonical_results.md` (IDs such as
`A3-10`, discrepancy numbers such as "disc. 11").

**Nothing listed here has been changed yet.** The website, the slides and the
READMEs still show the old values. Only the underlying data files were corrected:
`results/result_af3_*.json`, `results/sweep/af3_comparison.md` and the
`results/*.md` fixes for discrepancies 1, 2, 3 and 7.

## Where things live

| Short name | Path | Notes |
|---|---|---|
| Slides | `presentation/AlphaFold_on_Google_TPUs_Pazienza_Lorenzo_Ihab_El_Bani.pdf` | 5 slides. **Byte-identical copy** at `website/public/presentation/…pdf`: replace both |
| React site | `website/src/components/*.jsx`, `website/src/data/experiments.js` | What visitors see |
| Static fallback | `website/index.html` | `<meta>` descriptions and JSON-LD (lines 12–193) plus a no-JS / SEO copy of every section (lines 340+) |
| LLM transcript | `website/public/llms.txt` | Hand-written mirror of the page |
| READMEs | `README.md`, `results/sweep/README.md` | Public on GitHub |

A number usually has to change in **all four site layers**: component, static
fallback, `llms.txt` and README. Each item lists every occurrence found.

---

## 1. AlphaFold 3 timings (model inference instead of process wall-clock)

The old values were process wall-clock / 5. The new values are AF3's own
model-inference time / 5 (`A3-02`, `A3-05`, `A3-10`…`A3-12`). **The new AF3 figures
still include JIT compilation**, whereas AF2 is a warm second call.

**AF3-vs-AF2 ratios must be removed, not corrected** (items 1.3, 1.4, 1.6, 1.7;
disc. 15). Besides compilation:
- AF3 ran its default `--num_recycles 10` (11 trunk passes) against AF2's 0 (1 pass);
- AF3's per-sample figure spreads one trunk run over 5 diffusion samples.

Correcting only the recycle mismatch already flips the per-sample ratios below 1. No
single replacement number is defensible.

| # | What | Old | New | Occurrences |
|---|---|---|---|---|
| 1.1 | AF3 per sample, Colab CPU | 490.8s | **480.3s** | `AlphaFold3.jsx:13` (#af3 "Same-hardware · per sample" bars) · `index.html:570` · `llms.txt:219` |
| 1.2 | AF3 per sample, Colab T4 | 22.8s | **17.2s** | `AlphaFold3.jsx:21` · `index.html:576` · `llms.txt:220` |
| 1.3 | AF3 / AF2, CPU | 2.3× (2.31×) | **Remove.** Do not replace with 2.26× (disc. 15) | `AlphaFold3.jsx:14` · `index.html:571` · `llms.txt:219` · `README.md:336` · `results/sweep/README.md:195` |
| 1.4 | AF3 / AF2, GPU | 1.74× | **Remove.** Do not replace with 1.32× (disc. 15) | `AlphaFold3.jsx:22` · `index.html:577` · `llms.txt:220` · `README.md:336` · `results/sweep/README.md:196` |
| 1.5 | AF3 CPU→GPU speedup (`t_CPU / t_GPU`) | 21.5× | **27.9×**, as an AF3-only figure. Drop the "vs AF2's 16.2×" comparison: it compares different recycle counts and compilation treatment (disc. 15) | `AlphaFold3.jsx:65` · `index.html:582` · `llms.txt:222` · `README.md:337` |
| 1.6 | AF2-vs-AF3 bars (percentage of the AF3 bar) | CPU `af2Pct: 43`, GPU `af2Pct: 57` | **Remove the AF2-vs-AF3 bar pairs.** They draw the same non-interpretable ratio. Show AF3 per-sample times on their own (480.3s CPU, 17.2s T4), labelled "incl. JIT compilation, 10 recycles, 5 samples per call" | `AlphaFold3.jsx:15`, `AlphaFold3.jsx:23` |
| 1.7 | AF3-slower claim | "Same-hardware · per sample … AF3 is slower" and the "AF3 {ratio} AF2" labels | **Remove the claim and the labels.** Replace with: "AF2 and AF3 timings are not directly comparable here: AF3 ran 10 recycles (AF2: 0), includes JIT compilation, and produces 5 samples per call." | `AlphaFold3.jsx:61-66` · `index.html:557-582` · `llms.txt:216-222` · `README.md:335-338` |

The slides contain no AF3 timing numbers or AF3-vs-AF2 ratios; slide 5 only links to
`#af3`. Nothing needs removing there, but no future slide should add 2.26× or 1.32×.

---

## 2. "Same input / one script everywhere" is wrong outside the CPU/GPU/TPU baseline

What the repository shows (`paper/sections/methodology.md` §3.2, §5.1; disc. 12):
- **Toy sequence:** only the backend baseline (and the sweeps run with
  `spike_tpu_forward_pass.py` at 118 residues, plus ensemble and GSPMD) use the
  118-residue toy sequence.
- **Synthetic sequences:** **`vmap` batching, `pmap` multi-query and the scaling
  grid do not.** Every item there is a rotated repeat of the 20-letter amino-acid
  alphabet, even at 118 residues. The scaling grid covers lengths 100–1000, and the
  sequence-length sweep (60–500) is entirely synthetic.
- **Several scripts:** they are `spike_tpu_forward_pass.py`,
  `spike_batch_forward_pass.py`, `spike_pmap_forward_pass.py`,
  `spike_meshshard_forward_pass.py` and `spike_ensemble_shard_forward_pass.py`.
- **Not even one script version for the baseline:** CPU/GPU ran an uncommitted copy
  embedded in the Colab notebooks, and the TPU script version is unknown.

| # | Where | Current text | Correction |
|---|---|---|---|
| 2.1 | **Slide 1**, subtitle | "Same AlphaFold 2 inference, same 118-residue input." | "Same AlphaFold 2 model and 118-residue input for the CPU/GPU/TPU baseline; multi-chip and scaling experiments use synthetic sequences." |
| 2.2 | `Architecture.jsx:28` (#approach headline) · `index.html:344` (nav) | "One script, three backends" | e.g. "One model, three backends" |
| 2.3 | `Architecture.jsx:35-41` · `index.html:410-413` · `llms.txt:94-95,98` | "Every timing run calls `spike_tpu_forward_pass.py` with AlphaFold 2 model_3, 0 recycles, 118 residues. Only the accelerator changes." | Limit it to the backend baseline. Name the other four scripts. Say that sweeps vary length, recycles, model and precision. |
| 2.4 | `Problem.jsx:22-31` · `index.html:400-401` · `llms.txt:87` | "Same AlphaFold 2 forward pass, script, and shape (118 residues…)" | Keep "same model config and 118-residue shape" for the baseline. Drop "same script": the CPU/GPU notebook copy ≠ `src/`, and the TPU version is unknown. |
| 2.5 | `HeadlineResults.jsx:65` · `llms.txt:112` | "Identical code path" | Remove, or say "same timing structure" (disc. 12). |
| 2.6 | `README.md:17` · `README.md:189` | "same model, same script, same input shape" / "Identical workload" | Same correction as 2.3–2.4. |
| 2.7 | Experiment descriptions that imply the toy protein: seq-length `experiments.js:31`, batching `:99`, pmap `:110`, scaling law `:146` · `index.html:452-453, 494-496, 511-512, 529-530` · `llms.txt:140, 173, 182, 195` · **slide 4** (Protein₁…₈) · **slide 5** (scaling chart) | none | Add one line: "Synthetic sequences (rotated amino-acid alphabet), not the 118-residue toy protein." |

---

## 3. TPU headline: 8 chips allocated, 1 chip used (disc. 7)

The 451× and 27.8× figures come from a single-query run that uses only `TPU_0`.
Wherever they appear next to "8 chips", add "1 of 8 chips used".

| # | Where | Current | Correction |
|---|---|---|---|
| 3.1 | `Hero.jsx:46-51` (hero metric list) | "tpu-v5-lite-podslice · 2×4 · 8 chips" beside "451× … / 27.8× …" | "… · 8 chips (1 used)" |
| 3.2 | `Hero.jsx:79-80` (hero blurb) · `llms.txt:43` | "Steady-state AF2 on Stanford GKE TPU v5e-8 is 451× faster …" | "… on one v5e chip …" |
| 3.3 | `HeadlineResults.jsx:65` · `llms.txt:112` | "(2×4 lite podslice, 8 chips)" | "(2×4 lite podslice; 1 of 8 chips used)" |
| 3.4 | `index.html:12, 38, 55` (`<meta>` descriptions) · `index.html:193` (JSON-LD abstract) · `index.html:355-357` · `llms.txt:50` | "(tpu-v5-lite-podslice, 2×4, 8 chips) 0.47s (451× …)" | Add "1 chip used" |
| 3.5 | **Slide 2**, TPU row | "TPU v5e Lite 0.47 s · 451× faster than CPU · 27× faster than GPU" | Add "(1 of 8 chips)" |
| 3.6 | `README.md:197` (results table, Devices column) | "**8 chips**" | "**8 visible, 1 used**" (already fixed in `results/comparison.md:11`) |

## 4. Rounding error on slide 2

| # | Where | Old | New | Source |
|---|---|---|---|---|
| 4.1 | **Slide 2**, TPU row | "27× faster than GPU" | **27.8×** | 13.086 / 0.47 = 27.84 (`B-11`) |

---

## 5. First-call numbers: profiler overhead and the 76% figure (disc. 3, disc. 11)

| # | Where | Current | Correction | Source |
|---|---|---|---|---|
| 5.1 | **Slide 3**, "Cold start" | "27 s" | **27.8 s** (27.78). Add a footnote that the timed first call includes profiler-trace finalisation, which is 36–42 s on CPU/GPU and unmeasured on TPU. | `B-08`, disc. 11 |
| 5.2 | **Slide 3**, "FIRST PREDICTION" bar and callout | "76% JAX/XLA compilation / 24% other" · "~76% of first prediction → JAX/XLA compilation" | "~76% of the **traced `apply_fn` call** (12.55 of 16.56 s) is JAX tracing/compilation (`cache_miss` self time)". The 16.56 s span is not the 27.8 s cold start, so the bar must not be drawn over "first prediction". Also drop "24% other": `cache_miss` spans the whole 16.56 s, and the remaining 4.01 s is its child tracing frames (`_infer_params` → `trace_to_jaxpr`), not other work. | `TR-01…05`, disc. 3, **disc. 14** |
| 5.3 | `experiments.js:59` · `index.html:478` · `llms.txt:158` | "cache_miss … accounted for ~76% of the first predict call" | "… ~76% of the traced apply_fn call (16.56 s)" | disc. 3 |
| 5.4 | `experiments.js:61` · `llms.txt:157` | statLabel "of first call in cache_miss" | "of traced call in cache_miss" | disc. 3 |
| 5.5 | `experiments.js:63` · `llms.txt:159` | "12.55s self-time inside a 16.56s first predict" | "… inside the 16.56s traced apply_fn span (timed first predicts in other runs: 27–29s)" | disc. 3 |
| 5.6 | `README.md:26` · `README.md:64` · `README.md:236` | "~76% in pjit cache_miss" (of cold TPU calls) · "First TPU predict ≈ 16.56s" | Same wording as 5.3; "Traced apply_fn span ≈ 16.56s" | disc. 3 |
| 5.7 | `HeadlineResults.jsx:112` · `index.html:444` · `llms.txt:124` · `README.md:201, 238` | First/steady CPU **1.28×** | **~1.11×** after removing profiler finalisation ((271.98 − 36.00) / 212.113) | disc. 11 |
| 5.8 | `HeadlineResults.jsx:113` · `index.html:444` · `llms.txt:125` · `README.md:201, 251` | First/steady GPU **7.46×** ("Cold/warm ~7×") | **~4.25×** ((97.62 − 42.02) / 13.086) | disc. 11 |
| 5.9 | `HeadlineResults.jsx:114` · `index.html:444` · `llms.txt:126` · `README.md:201, 238, 252` | First/steady TPU **59.1×** | Keep the number, but mark it "may include profiler overhead, not measurable (no TPU logs)" | disc. 11 |
| 5.10 | `HeadlineResults.jsx:107-108` | "First predict includes XLA compile." | "First predict includes XLA compile and, in these runs, profiler-trace finalisation." | disc. 11 |

---

## 6. `pmap` per-chip memory (disc. 1)

| # | Where | Old | New |
|---|---|---|---|
| 6.1 | `experiments.js:110` · `llms.txt:182` | "Per-chip HBM was 445–469 MB" | "445–469 MB on chips 1–7, 644 MB on chip 0" (`MQ-08`) |

---

## 7. Slide 5 throughput chart and cost labels

| # | Where | Current | Problem | Correction |
|---|---|---|---|---|
| 7.1 | **Slide 5**, chart point labels and y-axis | "1,720 / 3,350 / 6,520 / 12,900", axis "Throughput (predictions / second)" | **Not found in any data file.** The highest measured throughput is 19.15 proteins/s (8 chips, length 100, `SC-04`). The label ratios (1 : 1.95 : 3.79 : 7.50) follow the fitted chip term, but no tested length reproduces the absolute values. The plotted dots also sit below their labels (e.g. the 1-chip dot is under 10³). | Plot measured values at a stated length. At length 100: **2.94 / 5.51 / 10.42 / 19.15 proteins/s**, or per hour **10,566 / 19,832 / 37,501 / 68,951**. State the length. The measured 8-over-1 speedup at length 100 is 6.53×, not the 7.5× the labels imply. No measured or fitted value in /s, /min or /h matches the labels; the nearest are 5–7% off and mix chip counts and lengths. See **disc. 13**. |
| 7.2 | **Slide 5**, "Flat price CPU $11.19 · GPU $1.27 · TPU $1.25" | Next to the scaling law | These are **baseline single-query costs** (TPU pod with 1 of 8 chips active, `C-08…C-10`). They are not the scaling-law flat-cost result (`SC-16…SC-19`). | Relabel, e.g. "Cost / 1k predictions, baseline path", or show `SC-16…19` instead. |
| 7.3 | `CostTakeaways.jsx:39-41` · `llms.txt:254` | "Multi-query pmap and single-query ensemble sharding (pmap + pmean) both fill the slice; after that, cost per prediction stays roughly constant as chip count grows." | The flat-cost result was measured only for **pmap multi-query** (scaling grid). The ensemble experiment has no throughput or cost measurement (`EN-09`). | "Multi-query pmap fills the slice; with it, cost per prediction stays roughly constant as chip count grows. Ensemble sharding also uses all 8 chips, but its cost was not measured." |

---

## 8. Docker and reproducibility claims (methodology §2.5, disc. 12)

No reported result was produced with the `Dockerfile`:
- **TPU:** the Jobs use `python:3.12-slim` and install packages at start-up.
- **CPU/GPU:** Colab notebooks.
- **Pins:** the Dockerfile's pins differ from both (`dm-haiku==0.0.12`,
  `tensorflow-cpu==2.16.1`, `jax` unpinned). Its "pinned to a fixed commit" comment
  is false.

| # | Where | Current | Correction |
|---|---|---|---|
| 8.1 | **Slide 4**, subtitle | "Built on Docker + Kubernetes (GKE, Kueue-managed TPU quota)" | "Kubernetes Jobs on GKE + Kueue (TPU); Google Colab notebooks (CPU/GPU)" |
| 8.2 | `Architecture.jsx:30` | "Docker for CPU/GPU images" | "Google Colab notebooks for CPU/GPU" |
| 8.3 | `index.html:349` (nav "Reproduce - repo + docker") · `index.html:636-639` · `llms.txt:269, 285-292` · `README.md:22, 275-285` | Docker presented as the reproduction path | State that the Dockerfile is not the environment that produced the results. Point to the notebooks and `configs/`, and note that the headline baselines cannot be reproduced exactly from the repo (disc. 12). |

---

## 9. Minor text fixes

| # | Where | Old | New |
|---|---|---|---|
| 9.1 | `Architecture.jsx:6` · `index.html:411` · `llms.txt:99` | `result_cpu.json` | `result_cpu-colab.json` (actual file name) |
| 9.2 | `llms.txt:300-302` (asset index) | "AF2 ubiquitin 3D fallback", "AF2 ubiquitin pLDDT chart", "AF2 interactive 3Dmol model" | "ESMFold ubiquitin …". The ubiquitin structure is ESMFold, not AlphaFold 2 (disc. 8). |

## 10. Claims that cannot be verified from the repo (no new value available)

| # | Where | Claim | Status |
|---|---|---|---|
| 10.1 | **Slide 2** · `Hero.jsx:31-32` · `llms.txt:48` · many "Intel Xeon (2 vCPU)" labels | Colab CPU is "Intel Xeon, 2 vCPU, 13.6 GB RAM" | Never printed by any notebook (methodology §1.1). Only `x86_64` is recorded. Keep only if confirmed from another source. |
| 10.2 | **Slides 2–5**, all TPU/GPU/CPU headline figures | Reproducible from the repo | Not reproducible as-is (disc. 12). Every headline value is also n = 1 per backend; the only repeat study is TPU-only (`RR-*`). |
