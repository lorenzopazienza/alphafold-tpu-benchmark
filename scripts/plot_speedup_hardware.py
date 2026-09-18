"""
Speedup figure for results_hardware.tex (T1-B deliverable, WORK_PLAN).

Steady-state time per call for each backend, as a bar chart on a log axis.
Above each bar: the measured time, and the speedup over the CPU baseline.
All numbers are the August 2026 baseline from
paper/data/canonical_results.md:

    B-03  CPU steady state        212.113 s
    B-06  GPU (T4) steady state    13.086 s
    B-09  TPU (v5e) steady state    0.47  s
    B-10  GPU over CPU             16.21x
    B-11  TPU over GPU             27.84x
    TPU over CPU is derived here as B-03 / B-09.

Do not edit a number here without updating canonical_results.md first.
Re-run the script; never edit the figure by hand.

Usage:  python3 scripts/plot_speedup_hardware.py
Output: figures/speedup_hardware.pdf (vector, for \\includegraphics)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

# ---- canonical numbers ------------------------------------------------------
ROWS = [
    # label,   detail,               seconds,  color (Okabe-Ito, colorblind-safe)
    ("CPU",    "Colab CPU runtime",   212.113,  "#0072B2"),   # B-03, blue (vCPU count is R-only, omitted)
    ("GPU",    "NVIDIA T4, Colab",     13.086,  "#E69F00"),   # B-06, orange
    ("TPU",    "one v5e chip of 8",     0.47,   "#009E73"),   # B-09, green
]
CPU_S = ROWS[0][2]
TPU_OVER_GPU = 27.84  # B-11

INK, MUTED, RULE = "#1F2328", "#6B7280", "#D9DCE1"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def fmt_seconds(s):
    return f"{s:,.3f} s" if s >= 1 else f"{s:.2f} s"


def main():
    fig, ax = plt.subplots(figsize=(5.2, 3.3))

    xs = range(len(ROWS))
    ax.bar(xs, [r[2] for r in ROWS], color=[r[3] for r in ROWS],
           width=0.56, zorder=3)

    for x, (label, detail, s, color) in zip(xs, ROWS):
        # measured value, right above the bar
        ax.annotate(fmt_seconds(s), xy=(x, s), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9.5, color=INK, fontweight="bold")
        # speedup over the CPU baseline, one line higher
        if label == "CPU":
            note = "baseline"
        else:
            r = CPU_S / s
            note = f"{r:.1f}$\\times$ faster than CPU" if r < 100 \
                else f"{r:.0f}$\\times$ faster than CPU"
        ax.annotate(note, xy=(x, s), xytext=(0, 17),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=MUTED)
        # x labels: backend in ink, hardware detail muted
        ax.text(x, -0.06, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color=INK,
                fontweight="bold")
        ax.text(x, -0.14, detail, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5, color=MUTED)

    # the second ratio quoted in the text (B-11), under the TPU bar
    ax.text(2, -0.23, f"{TPU_OVER_GPU:.1f}$\\times$ faster than GPU",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7.5, color=ROWS[2][3])

    # y axis: log, plain-number ticks
    ax.set_yscale("log")
    ax.set_ylim(0.1, 3000)
    ticks = [0.1, 1, 10, 100, 1000]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10", "100",
                                                 "1000"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Steady-state time per call (s, log scale)",
                  fontsize=8.5, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=3, width=0.6)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 2.6)
    ax.grid(axis="y", color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)

    fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.26)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "speedup_hardware.pdf")
    fig.savefig(out)
    print(f"Wrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
