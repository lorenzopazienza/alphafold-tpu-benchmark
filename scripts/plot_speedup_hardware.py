"""
Speedup figure for results_hardware.tex (T1-B deliverable, WORK_PLAN).

Steady-state single-call timing per backend, log scale, with the three
pairwise speedup ratios annotated. All numbers are the August 2026
baseline from paper/data/canonical_results.md (rows B-03, B-06, B-09,
B-10, B-11; B-10/B-11 also appear as 16.2x / 27.8x in
results/comparison.md). Do not edit the numbers here without updating
canonical_results.md first - this script is meant to be re-run, not
hand-edited to change a value.

Usage: python3 scripts/plot_speedup_hardware.py
Output: figures/speedup_hardware.pdf (vector, for \\includegraphics in
the paper) and figures/speedup_hardware.png (quick preview only).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# --- Canonical numbers (canonical_results.md, master table, status M) ---
BACKENDS = ["CPU", "GPU (T4)", "TPU (v5e, 1 chip)"]
STEADY_STATE_S = [212.113, 13.086, 0.47]   # B-03, B-06, B-09

# Pairwise speedups already in canonical_results.md (B-10, B-11) plus the
# TPU-over-CPU figure quoted in the paper text (derived, not a separate
# canonical row: 212.113 / 0.47).
SPEEDUP_GPU_OVER_CPU = 16.21   # B-10
SPEEDUP_TPU_OVER_GPU = 27.84   # B-11
SPEEDUP_TPU_OVER_CPU = 212.113 / 0.47

# Okabe-Ito colorblind-safe palette, fixed categorical order (never cycled).
COLORS = ["#0072B2", "#E69F00", "#009E73"]      # blue, orange, green
HATCHES = ["", "///", "xxx"]                     # grayscale/print safety

def main():
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    x = range(len(BACKENDS))
    bars = ax.bar(
        x, STEADY_STATE_S,
        color=COLORS, hatch=HATCHES,
        edgecolor="black", linewidth=0.8, width=0.6,
    )

    ax.set_yscale("log")
    ax.set_ylabel("Steady-state time per call (s, log scale)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(BACKENDS)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Direct value labels on each bar (selective, not a number-everywhere
    # chart: exactly one label per bar, the mark's own value).
    for rect, val in zip(bars, STEADY_STATE_S):
        ax.annotate(
            f"{val:g} s",
            xy=(rect.get_x() + rect.get_width() / 2, val),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    # Pairwise speedup annotations as bracketed callouts above the bars,
    # so the ratios in the text (Section results_hardware) are visible
    # on the figure itself rather than only in prose. Heights are chosen
    # first, then the y-limit is set to clear the highest one - no LaTeX
    # \ref or \cite here, matplotlib cannot resolve those; cross-references
    # belong in the \caption in results_hardware.tex, not in the image.
    def bracket(x0, x1, y, label):
        ax.plot([x0, x0, x1, x1], [y * 0.85, y, y, y * 0.85],
                 color="0.35", linewidth=0.9)
        ax.text((x0 + x1) / 2, y * 1.1, label,
                 ha="center", va="bottom", fontsize=8.5, color="0.25")

    top_bracket_y = STEADY_STATE_S[0] * 9.0
    bracket(0, 1, STEADY_STATE_S[0] * 1.6, f"{SPEEDUP_GPU_OVER_CPU:.1f}$\\times$")
    bracket(1, 2, STEADY_STATE_S[0] * 3.4, f"{SPEEDUP_TPU_OVER_GPU:.1f}$\\times$")
    bracket(0, 2, top_bracket_y, f"{SPEEDUP_TPU_OVER_CPU:.0f}$\\times$")
    ax.set_ylim(top=top_bracket_y * 1.6)

    ax.set_title(
        "AlphaFold2 steady-state inference, single call",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "speedup_hardware.pdf"))
    fig.savefig(os.path.join(out_dir, "speedup_hardware.png"), dpi=200)
    print("Wrote figures/speedup_hardware.pdf and .png")

if __name__ == "__main__":
    main()
