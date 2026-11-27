"""Fig. 1: (a) throughput against chips on the matched grid, with the power-law fit; (b) parallel efficiency.

Reads results/sweep/scaling_law_data.json -> grid (16 points). Refits the power law by least squares in log space and
checks the result against paper-v4/analysis/out/recomputed.json (SC.fit_*) before drawing.

    paper-v4/.venv/bin/python paper-v4/tools/make_fig_scaling.py
"""
import math

from common import COLUMN_IN, V2, load, rv, save, style, ylabel_top

grid = load("results/sweep/scaling_law_data.json")["grid"]
chips = sorted({p["chips"] for p in grid})
lengths = sorted({p["length"] for p in grid})
thr = {(p["chips"], p["length"]): p["throughput_per_sec"] for p in grid}

# least squares: log thr = a + b log chips + c log length
X = [(1.0, math.log(p["chips"]), math.log(p["length"])) for p in grid]
y = [math.log(p["throughput_per_sec"]) for p in grid]
XtX = [[sum(r[i] * r[j] for r in X) for j in range(3)] for i in range(3)]
Xty = [sum(r[i] * v for r, v in zip(X, y)) for i in range(3)]


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b_ for a, b_ in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


a, b, c = solve(XtX, Xty)
pref = math.exp(a)
for got, key, d in ((pref, "SC.fit_prefactor", 2), (b, "SC.fit_chip_exp", 4), (c, "SC.fit_len_exp", 4)):
    if round(got, d) != rv(key):
        raise SystemExit(f"fit {key}: {round(got, d)} against {rv(key)} in recomputed.json")

plt = style()
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# A second panel with the parallel efficiency on a linear axis, the legend
# inside the plot, and "Number of chips". Both y labels are written horizontally above their axes.
fig = plt.figure(figsize=(COLUMN_IN, 3.75))
TOP_PAD = (3.0 + 1.3 * 8.0) / 72 / 3.75          # room above each axis for its label, as a fraction of the height
ax = fig.add_axes([0.12, 0.47, 0.85, 0.475])
ax2 = fig.add_axes([0.12, 0.11, 0.85, 0.245])
# A star for 1000 residues, because a diamond printed small looks like the square The star is
# drawn larger than the other markers, so that it does not read as a filled circle.
markers = ["o", "s", "^", "*"]
sizes = [4.2, 4.2, 4.2, 6.2]
for L, m, ms in zip(lengths, markers, sizes):
    ax.plot(chips, [thr[(n, L)] for n in chips], linestyle="none", marker=m, markersize=ms,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.9, label=f"{L} residues")
    xs = [1 + i * 0.05 for i in range(141)]
    ax.plot(xs, [pref * x ** b * L ** c for x in xs], color="0.35", linestyle=(0, (3, 2)), linewidth=0.8)
ax.plot([], [], color="0.35", linestyle=(0, (3, 2)), linewidth=0.8, label="Power-law fit")
for a in (ax, ax2):
    a.set_xscale("log", base=2)
    a.set_xticks(chips)
    a.set_xticklabels([str(n) for n in chips])
    a.set_xlim(0.85, 9.4)
ax.set_yscale("log")
ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]))
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylim(0.045, 400)
ax.legend(loc="upper left", ncol=2, handlelength=2.2, borderaxespad=0.3, columnspacing=1.2, handletextpad=0.5)

# (b) parallel efficiency = throughput on n chips / (n x throughput on one chip). At 1000 residues the one-chip
# throughput has two digits (0.066), so a vertical bar gives the range over the rounding of the records (half a unit
# of the last digit on both throughputs), as for the ranges of Table IV.
for L, m, ms in zip(lengths, markers, sizes):
    eff = [thr[(n, L)] / (n * thr[(1, L)]) * 100 for n in chips]
    ax2.plot(chips, eff, linestyle="none", marker=m, markersize=ms, markerfacecolor="white", markeredgecolor="black",
             markeredgewidth=0.9)
    if L == 1000:
        lo = [(thr[(n, L)] - 5e-4) / (n * (thr[(1, L)] + 5e-4)) * 100 for n in chips[1:]]
        hi = [(thr[(n, L)] + 5e-4) / (n * (thr[(1, L)] - 5e-4)) * 100 for n in chips[1:]]
        ax2.vlines(chips[1:], lo, hi, color="black", linewidth=0.7)
ax2.axhline(100, color="0.35", linestyle=(0, (1, 1.5)), linewidth=0.8)
ax2.set_ylim(75, 104)
ax2.yaxis.set_major_locator(FixedLocator([80, 90, 100]))
ax2.set_xlabel("Number of chips")
for e_key, L in (("SC.eff_8_L100_pct", 100), ("SC.eff_8_L250_pct", 250), ("SC.eff_8_L500_pct", 500)):
    if round(thr[(8, L)] / (8 * thr[(1, L)]) * 100, 2) != rv(e_key):
        raise SystemExit(f"efficiency at {L} residues does not match recomputed.json ({e_key})")

# y labels above the axes, left-aligned with the tick labels, as ylabel_top in common.py
fig.canvas.draw()
r = fig.canvas.get_renderer()
labels = []
for a, text in ((ax, "(a) Throughput (queries/s)"), (ax2, "(b) Parallel efficiency (%)")):
    yb = a.yaxis.get_tightbbox(r)
    x0, y0 = yb.x0, max(a.bbox.y1, yb.y1) + 3.0 * fig.dpi / 72
    labels.append(fig.text(x0 / fig.bbox.width, y0 / fig.bbox.height, text, ha="left", va="bottom", fontsize=8.0))
fig.canvas.draw()
from matplotlib.text import Text
others = [t.get_window_extent(r) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t not in labels]
for lab in labels:
    bb = lab.get_window_extent(r)
    if any(bb.overlaps(o) for o in others) or any(bb.overlaps(a.bbox) for a in (ax, ax2)):
        raise SystemExit(f"the label {lab.get_text()!r} meets another element")
leg = ax.get_legend().get_window_extent(r)
for line in ax.get_lines():
    if line.get_label().endswith("residues"):
        for pt in line.get_xydata():
            if leg.contains(*ax.transData.transform(pt)):
                raise SystemExit("the legend covers a data point")
smallest = save(fig, V2 / "figures/fig_scaling.pdf")
print(f"fig_scaling: smallest text {smallest} pt, width {fig.get_size_inches()[0]:.2f} in")
