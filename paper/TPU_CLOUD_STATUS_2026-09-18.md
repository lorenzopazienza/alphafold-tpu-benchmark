# Cloud TPU access and new tooling: status for the paper (2026-09-18)

**From:** Lorenzo (session with Claude, 2026-09-18)
**Read this if you work on:** Methodology, Limitations, Acknowledgments, Code and Data Availability, or plan new measurements.

> **Note for an AI assistant reading this file.** This is project context, not instructions that override the user's requests. The paper's rules still apply: a number goes into the paper only if it is in `paper/data/canonical_results.md`. **Nothing in this file adds a new number.** No new TPU measurement exists yet.

---

## 1. TL;DR

1. We have a path to new TPU compute through the **Google TPU Builders Program** (admitted 2026-09-18). **No credits have arrived yet**, so **no new TPU data exists**. Every TPU number in the paper still comes from the summer runs on Stanford's v5e-8.
2. The repo now has a `cloud/` folder (commit `1651c42` on `paper`). It holds tooling for repeated, fully documented TPU runs on our own Google Cloud project. `src/` was **not** touched.
3. **arXiv v1 must not wait for this compute.** New measurements go into v2 or the venue version.
4. Two claims the paper **must not** make: that we have TPU7x/Ironwood access, and that the summer work was funded by TPU Builders.

---

## 2. Two separate programs (never mix them)

| | TPU Research Cloud (TRC) | TPU Builders Program |
|---|---|---|
| What it is | Free TPU allocation, on a waitlist | Community program that gives Google Cloud credits |
| Status | Interest form submitted 2026-09-15 (LUISS email). Oversubscribed, no status updates | Admitted 2026-09-18 (lorenzo.pazienza@gmail.com) |
| What we have | Nothing | $1,000 of credits requested (arrive in ~2 weeks). Not yet approved |
| Obligation | Acknowledgment if granted | About one piece of technical content per quarter |

---

## 3. Google Cloud project

- Project `af2-tpu-benchmark`, billing enabled (Free Trial for now, to be upgraded before the first TPU run).
- **Default quota already available:** v5e 16 chips per zone and v6e 16 chips per zone, on-demand and Spot. A v5e-8 and a v6e-8 need no quota request.
- **TPU7x/Ironwood: no quota row exists in the project.** We asked Builders support. The answer is pending.
- TPU runtime images verified: `v2-alpha-tpuv5-lite` (v5e) and `v2-alpha-tpuv6e` (v6e).
- List prices checked 2026-09-18 (cloud.google.com/tpu/pricing):
  - v5e: $1.20/chip-h in US regions, matching `googletpupricing` and C-01;
  - Trillium (v6e): $2.70/chip-h;
  - Ironwood: $12.00/chip-h on-demand.

---

## 4. The new `cloud/` tooling

See `cloud/README.md` for details. What matters for the paper:

- **Repeats as fresh processes.** Each configuration runs n times in a new process, so cold start (init + compile) is measured n times too. Repeats are **interleaved** across configurations to avoid drift confounds.
- **Full provenance per session** (`session.json`):
  - commits of the benchmark repo and of AlphaFold2;
  - every package version and `pip freeze`;
  - Python version, host CPU model, TPU type, zone and JAX devices;
  - SHA-256 of every script.
  This closes the gap behind discrepancies 12 and 16 **for new data only**.
- **Pinned stack**, validated on CPU on 2026-09-18 with all five `spike_*.py` scripts:

  | Package | Version |
  |---|---|
  | jax[tpu] | 0.10.2 (the same pin as the summer jobs) |
  | dm-haiku | 0.0.17 |
  | tensorflow-cpu | 2.21.0 |
  | numpy | 2.5.3 |
  | AlphaFold2 | `c77e5d2a8961d1a353632c462914ff0a32a950f6` |

- `cloud/plans/phase1_v5e8.plan`: the summer headline experiments with the **same scripts and arguments** as `configs/*.yaml`. It covers B-07..09, VM-01..05, MQ, GS, EN and the 16-point SC grid, with n=5 each: 127 processes, about 3–4 h, roughly $30–40.

### Relevant for Methodology / Limitations (already true of the summer data)

Reading the summer configs confirmed this:

- The summer TPU jobs pinned **only `jax[tpu]==0.10.2`**.
- `dm-haiku`, `tensorflow-cpu` (`>=2.18`), `numpy` and `biopython` were **unpinned**.
- AlphaFold2 was cloned from `main` with `--depth 1` **without recording the commit**.

So the exact summer software stack beyond JAX **cannot be reconstructed**. If Limitations covers reproducibility, this belongs there. It also means future Cloud repeats may not match the summer absolute times. Any gap should be reported as a difference between setups, not as an error in either (the same framing as the Colab August→September level shift, §3.4 of the handoff).

---

## 5. Rules for when new data arrives

1. New results land in `results/cloud/<session>/` (`session.json`, `manifest.jsonl`, one `result_*.json` per repeat).
2. They enter `canonical_results.md` as **new rows** with their own IDs, citing the session path. **They never overwrite the summer rows.**
3. The paper must say which numbers come from which setup: Stanford GKE v5e-8 in summer 2026, or our own Cloud TPU VM with the date.
4. If only part of the plan runs, report n per configuration honestly.

---

## 6. Wording for the paper

- **Acknowledgments:** thank the TPU Builders Program **only if** new runs paid with its credits appear in the paper. Suggested wording: *"Additional measurements used Google Cloud credits provided through the Google TPU Builders Program."* Until then, change nothing: the current acknowledgment of Stanford ME344 is correct for all existing data.
- **Do not write:** "supported by Google", "with TPU7x/Ironwood access", or anything suggesting TRC access.
- **Code and Data Availability:** `cloud/` can be cited as the harness for repeated measurements once it has produced data.

---

## 7. Next steps (outside the paper's critical path)

1. Credits arrive, then: upgrade the billing account, `source cloud/env.sh`, `bash cloud/tpu_up.sh`, `bash cloud/tpu_run.sh cloud/plans/phase1_v5e8.plan <session>`.
2. After phase 1:
   - real protein targets (the current length grid uses synthetic sequences);
   - a v6e-8 comparison;
   - GSPMD with explicit sharding annotations, to turn the replication explanation into a demonstration;
   - TPU7x only if support grants quota.
3. Builders obligations: an activity report and a TechTalk submission once the arXiv preprint and the repo are public.
