"""
Figures 2-5 for the paper, built only from files in results/ and from
paper/data/canonical_results.md. Every number below cites its canonical
row or source file. Do not edit a number here without updating
canonical_results.md first; re-run the script, never edit a figure by hand.

Usage:  python3 scripts/plot_paper_figures.py  [--repo PATH]
Output: figures/parallelism_chips.pdf, figures/scaling_grid.pdf,
        figures/coldstart_regions.pdf, figures/session_variability.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

# ---- style (matches plot_speedup_hardware.py) --------------------------------
INK, MUTED, RULE, FAINT = "#1F2328", "#6B7280", "#D9DCE1", "#EEF0F3"
BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"  # Okabe-Ito
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
})


def clean(ax, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3, width=0.6)
    ax.grid(axis="y", color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=9, color=INK, pad=6)


# =============================================================================
# Figure: parallelism on the 8-chip slice
# =============================================================================
def fig_parallelism(repo, out_dir):
    batching = json.load(open(os.path.join(repo, "results/sweep/batching_sweep.json")))
    sharding = json.load(open(os.path.join(repo, "results/sweep/sharding.json")))
    chips = json.load(open(os.path.join(repo, "results/sweep/chip_visibility_sweep.json")))
    ensemble = json.load(open(os.path.join(repo, "results/sweep/ensemble_shard.json")))

    pm = sharding["pmap_data_parallelism"]
    gs = sharding["mesh_auto_sharding"]
    vis8 = [c for c in chips if c["chips_visible"] == 8][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.75),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # --- (a) throughput per configuration --------------------------------------
    # VM-01..04: vmap B=1,2,4,8 on one chip; MQ-05: pmap, 8 queries on 8 chips.
    labels = [f"vmap\nB={b['batch_size']}" for b in batching] + ["pmap\n8 queries,\n8 chips"]
    vals = [b["throughput_proteins_per_sec"] for b in batching] + [pm["throughput_proteins_per_sec"]]
    colors = [MUTED] * len(batching) + [GREEN]
    xs = range(len(vals))
    ax1.bar(xs, vals, color=colors, width=0.62, zorder=3)
    for x, v in zip(xs, vals):
        ax1.annotate(f"{v:.2f}", xy=(x, v), xytext=(0, 3 if x == len(vals) - 1 else 12),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=7.5, color=INK)
    ax1.axhline(vals[0], color=INK, linewidth=0.6, linestyle=(0, (3, 2)), zorder=2)
    ax1.text(-0.4, vals[0] + 0.45, "single query, one chip",
             ha="left", va="bottom", fontsize=6.8, color=MUTED)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels, fontsize=7.2, color=INK)
    ax1.set_ylim(0, 17)
    clean(ax1, "Throughput (proteins / s)")
    ax1.tick_params(axis="x", length=0)
    panel_label(ax1, "(a) Throughput by JAX transformation")

    # --- (b) per-chip HBM footprint ----------------------------------------------
    rows = [
        ("Default, 1 query\n(CV-09)",
         [d["bytes_in_use"] / 1e6 for d in vis8["memory_stats_per_device"]], BLUE, None),
        ("Auto-mesh partitioning,\n1 query (GS-02)",
         None, PINK, gs["run_1"]["memory_per_chip_mb"]),
        ("pmap, 8 queries\n(MQ-08)",
         pm["memory_per_chip_mb"], GREEN, None),
        ("pmap + pmean,\n8-member ensemble\n(EN-06)",
         [d["bytes_in_use_mb"] for d in ensemble["memory_stats_per_device_mb"]], ORANGE, None),
    ]
    n_rows = len(rows)
    ymax = 900
    for r, (name, per_chip, color, scalar) in enumerate(rows):
        base = (n_rows - 1 - r) * 1.0
        # a faint band per row
        ax2.axhspan(base, base + 0.92, color=FAINT if r % 2 else "white", zorder=0, linewidth=0)
        ax2.text(-0.75, base + 0.46, name, ha="right", va="center", fontsize=7,
                 color=INK)
        if per_chip is not None:
            for i, mb in enumerate(per_chip):
                h = 0.82 * mb / ymax
                ax2.bar(i, h, bottom=base + 0.05, width=0.7, color=color, zorder=3)
                if mb > 0:
                    ax2.text(i, base + 0.05 + h + 0.03, f"{mb:.0f}", ha="center",
                             va="bottom", fontsize=6, color=INK)
                else:
                    ax2.text(i, base + 0.12, "0", ha="center", va="bottom",
                             fontsize=6, color=MUTED)
        else:
            h = 0.82 * scalar / ymax
            ax2.hlines(base + 0.05 + h, -0.35, 7.35, color=color, linewidth=1.4,
                       linestyle=(0, (4, 2)), zorder=3)
            ax2.text(3.5, base + 0.05 + h + 0.04,
                     f"{scalar} MB per chip, one recorded figure (no per-device list)",
                     ha="center", va="bottom", fontsize=6.3, color=INK)
    ax2.set_xlim(-0.6, 7.6)
    ax2.set_ylim(0, n_rows)
    ax2.set_xticks(range(8))
    ax2.set_xticklabels([f"TPU_{i}" for i in range(8)], fontsize=7, color=MUTED)
    ax2.set_yticks([])
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["bottom"].set_color(MUTED)
    ax2.tick_params(axis="x", length=0)
    ax2.grid(False)
    panel_label(ax2, "(b) HBM in use per chip (MB)")

    fig.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.2, wspace=0.55)
    out = os.path.join(out_dir, "parallelism_chips.pdf")
    fig.savefig(out)
    print("Wrote", out)


# =============================================================================
# Figure: scaling grid and fit
# =============================================================================
def fig_scaling(repo, out_dir):
    grid = json.load(open(os.path.join(repo, "results/sweep/scaling_law_data.json")))["grid"]
    # SC-05..08: fitted law (from canonical_results.md / scaling_law.md)
    A, a, b = 4527.77, 0.963, -1.572
    lengths = sorted({g["length"] for g in grid})
    chips_all = sorted({g["chips"] for g in grid})
    colors = dict(zip(lengths, [BLUE, ORANGE, GREEN, PINK]))
    markers = dict(zip(lengths, ["o", "s", "^", "D"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.75),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- (a) measured throughput vs chips, log-log, with the fit ------------------
    for L in lengths:
        pts = sorted([g for g in grid if g["length"] == L], key=lambda g: g["chips"])
        xs = [p["chips"] for p in pts]
        ys = [p["throughput_per_sec"] for p in pts]
        ax1.plot(xs, [A * c ** a * L ** b for c in xs], color=colors[L], linewidth=1.0,
                 linestyle=(0, (3, 2)), zorder=2)
        ax1.plot(xs, ys, marker=markers[L], color=colors[L], linewidth=0, markersize=5,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax1.text(8.6, ys[-1], f"{L} residues", ha="left", va="center", fontsize=7,
                 color=INK)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.85, 14)
    ax1.xaxis.set_major_locator(FixedLocator(chips_all))
    ax1.xaxis.set_major_formatter(FixedFormatter([str(c) for c in chips_all]))
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_ylim(0.04, 30)
    ax1.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.5, 1, 5, 10, 20]))
    ax1.yaxis.set_major_formatter(FixedFormatter(["0.05", "0.1", "0.5", "1", "5", "10", "20"]))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_xlabel("TPU chips in use", fontsize=8, color=MUTED)
    clean(ax1, "Throughput (proteins / s)")
    ax1.text(0.9, 21, "markers: measured\ndashed: fitted law", fontsize=6.5,
             color=MUTED, ha="left", va="top")
    panel_label(ax1, "(a) Measured grid and fitted power law")

    # --- (b) fitted / measured ratio per point ------------------------------------
    width = 0.19
    for j, L in enumerate(lengths):
        pts = sorted([g for g in grid if g["length"] == L], key=lambda g: g["chips"])
        for i, p in enumerate(pts):
            ratio = (A * p["chips"] ** a * L ** b) / p["throughput_per_sec"]
            x = i + (j - 1.5) * width
            ax2.bar(x, ratio - 1, bottom=1, width=width * 0.92, color=colors[L], zorder=3)
    ax2.axhline(1, color=INK, linewidth=0.7, zorder=4)
    ax2.set_xticks(range(len(chips_all)))
    ax2.set_xticklabels([f"{c} chip{'s' if c > 1 else ''}" for c in chips_all],
                        fontsize=7.5, color=INK)
    ax2.set_ylim(0.7, 1.4)
    ax2.set_yticks([0.75, 1.0, 1.25])
    ax2.set_yticklabels(["0.75", "1.00", "1.25"])
    clean(ax2, "Fitted / measured throughput")
    ax2.tick_params(axis="x", length=0)
    ax2.text(-0.5, 1.37, "fit over-predicts", fontsize=6.5, color=MUTED, va="top")
    ax2.text(-0.5, 0.72, "fit under-predicts", fontsize=6.5, color=MUTED, va="bottom")
    # legend for panel b (colors = lengths)
    from matplotlib.patches import Patch
    ax2.legend([Patch(color=colors[L]) for L in lengths],
               [f"{L} res." for L in lengths], fontsize=6.5, frameon=False,
               loc="upper right", ncol=2, handlelength=1.0, columnspacing=0.8)
    panel_label(ax2, "(b) Where the fit misses")

    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.2, wspace=0.35)
    out = os.path.join(out_dir, "scaling_grid.pdf")
    fig.savefig(out)
    print("Wrote", out)


# =============================================================================
# Figure: cold-start regions and the compilation cache
# =============================================================================
def fig_coldstart(repo, out_dir):
    base = {}
    for name, f in [("CPU", "result_cpu-colab.json"), ("GPU (T4)", "result_gpu-t4.json"),
                    ("TPU (1 chip)", "result_tpu-v5e-podslice.json")]:
        d = json.load(open(os.path.join(repo, "results", f)))
        base[name] = (d["init_params_seconds"], d["first_predict_compile_and_run_seconds"],
                      d["second_predict_steady_state_seconds"])
    # discrepancy 11: profiler teardown inside the first-call timer (CPU 36.00 s, GPU 42.02 s)
    overhead = {"CPU": 36.00, "GPU (T4)": 42.02, "TPU (1 chip)": None}
    cache = json.load(open(os.path.join(repo, "results/sweep/compilation_cache.json")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.75),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # --- (a) three timed regions per backend, log scale ---------------------------
    regions = ["init_params", "first predict", "steady state"]
    rcol = [BLUE, ORANGE, GREEN]
    width = 0.25
    names = list(base)
    for i, name in enumerate(names):
        for j, (val, col) in enumerate(zip(base[name], rcol)):
            x = i + (j - 1) * width
            oh = overhead[name] if j == 1 else None
            if oh:
                ax1.bar(x, val - oh, width=width * 0.9, color=col, zorder=3)
                ax1.bar(x, oh, bottom=val - oh, width=width * 0.9, color=col, alpha=0.35,
                        zorder=3, linewidth=0)
            else:
                ax1.bar(x, val, width=width * 0.9, color=col, zorder=3)
            label = f"{val:,.1f}" if val >= 10 else (f"{val:.2f}" if val < 1 else f"{val:.1f}")
            ax1.annotate(label, xy=(x, val), xytext=(0, 2.5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=6.6, color=INK)
    ax1.set_yscale("log")
    ax1.set_ylim(0.2, 1200)
    ax1.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax1.yaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1000"]))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=8, color=INK)
    clean(ax1, "Seconds (log scale)")
    ax1.tick_params(axis="x", length=0)
    from matplotlib.patches import Patch
    ax1.legend([Patch(color=c) for c in rcol] + [Patch(color=ORANGE, alpha=0.35)],
               regions + ["profiler teardown\ninside the timer"], fontsize=6.5,
               frameon=False, loc="upper right", ncol=2, handlelength=1.0,
               columnspacing=0.8)
    panel_label(ax1, "(a) Timed regions, August baselines")

    # --- (b) TPU: cold vs warm persistent compilation cache -------------------------
    cold = [cache["cold"]["init_params_seconds"], cache["cold"]["first_predict_seconds"]]
    warm = [cache["warm"]["init_params_seconds"], cache["warm"]["first_predict_seconds"]]
    xs = [0, 1]
    w = 0.34
    ax2.bar([x - w / 2 for x in xs], cold, width=w * 0.92, color=MUTED, zorder=3)
    ax2.bar([x + w / 2 for x in xs], warm, width=w * 0.92, color=GREEN, zorder=3)
    for x, c, wv in zip(xs, cold, warm):
        ax2.annotate(f"{c:.1f}", xy=(x - w / 2, c), xytext=(0, 2.5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.8, color=INK)
        ax2.annotate(f"{wv:.1f}", xy=(x + w / 2, wv), xytext=(0, 2.5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.8, color=INK)
        ax2.text(x, max(c, wv) + 5.5, f"{c / wv:.2f}$\\times$", ha="center", va="bottom",
                 fontsize=7.5, color=GREEN, fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["init_params", "first predict"], fontsize=8, color=INK)
    ax2.set_ylim(0, 50)
    clean(ax2, "Seconds")
    ax2.tick_params(axis="x", length=0)
    ax2.legend([Patch(color=MUTED), Patch(color=GREEN)], ["cold cache", "warm cache"],
               fontsize=6.8, frameon=False, loc="upper right", handlelength=1.0)
    panel_label(ax2, "(b) TPU, persistent compilation cache")

    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.14, wspace=0.3)
    out = os.path.join(out_dir, "coldstart_regions.pdf")
    fig.savefig(out)
    print("Wrote", out)


# =============================================================================
# Figure: cross-session variability of the Colab baselines
# =============================================================================
def fig_variability(repo, out_dir):
    def noprofile(folder, tag):
        f = os.path.join(repo, "results/repro", folder,
                         f"result_{tag}_model_3_len118_recycle0_float32_noprofile.json")
        d = json.load(open(f))
        return d["steady_state_mean_seconds"], d["steady_state_stdev_seconds"], d["num_steady_state_runs"]

    cpu_aug = json.load(open(os.path.join(repo, "results/result_cpu-colab.json")))["second_predict_steady_state_seconds"]
    gpu_aug = json.load(open(os.path.join(repo, "results/result_gpu-t4.json")))["second_predict_steady_state_seconds"]
    cpu = [("Aug 8", cpu_aug, 0.0, 1)]
    for folder, lab in [("2026-09-15_cpu-colab", "Sep 15"), ("2026-09-16_cpu-colab", "Sep 16"),
                        ("2026-09-17_cpu-colab", "Sep 17")]:
        m, s, n = noprofile(folder, "cpu-colab-repro")
        cpu.append((lab, m, s, n))
    gpu = [("Aug 8", gpu_aug, 0.0, 1)]
    m, s, n = noprofile("2026-09-15_gpu-t4", "gpu-t4-repro")
    gpu.append(("Sep 15", m, s, n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, data, color, title, unit in [
        (ax1, cpu, BLUE, "(a) CPU (Colab), steady state", "s"),
        (ax2, gpu, ORANGE, "(b) GPU (Colab T4), steady state", "s"),
    ]:
        xs = range(len(data))
        for x, (lab, m, s, n) in zip(xs, data):
            ax.errorbar(x, m, yerr=s if n > 1 else None, fmt="o", color=color, markersize=6,
                        capsize=3, elinewidth=1, markeredgecolor="white", markeredgewidth=0.6,
                        zorder=4)
            txt = f"{m:.1f} {unit}" if m >= 10 else f"{m:.2f} {unit}"
            txt += f"\n(n={n}" + (f", sd {s:.2f})" if n > 1 else ")")
            ax.annotate(txt, xy=(x, m), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.6, color=INK)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([d[0] for d in data], fontsize=8, color=INK)
        ax.set_xlim(-0.6, len(data) - 0.4)
        clean(ax, "Seconds per call")
        ax.tick_params(axis="x", length=0)
        panel_label(ax, title)
    ax1.set_ylim(0, 450)
    ax2.set_ylim(0, 17)
    ax1.text(0.5, 0.03, "same benchmark script in the three September sessions",
             transform=ax1.transAxes, ha="center", va="bottom", fontsize=6.5, color=MUTED)

    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.14, wspace=0.3)
    out = os.path.join(out_dir, "session_variability.pdf")
    fig.savefig(out)
    print("Wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.repo, "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig_parallelism(args.repo, out_dir)
    fig_scaling(args.repo, out_dir)
    fig_coldstart(args.repo, out_dir)
    fig_variability(args.repo, out_dir)


if __name__ == "__main__":
    main()
