"""Fig. 1: throughput against chips on the matched grid, with the power-law fit.

Reads results/sweep/scaling_law_data.json -> grid (16 points). Refits the power law by least squares in log space and
checks the result against paper-v3/analysis/out/recomputed.json (SC.fit_*) before drawing.

    paper-v3/.venv/bin/python paper-v3/tools/make_fig_scaling.py
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
fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.35))
# A star for 1000 residues, because a diamond printed small looks like the square. The star is
# drawn larger than the other markers, so that it does not read as a filled circle.
markers = ["o", "s", "^", "*"]
sizes = [4.2, 4.2, 4.2, 6.2]
for L, m, ms in zip(lengths, markers, sizes):
    ax.plot(chips, [thr[(n, L)] for n in chips], linestyle="none", marker=m, markersize=ms,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.9, label=f"{L} residues")
    xs = [1 + i * 0.05 for i in range(141)]
    ax.plot(xs, [pref * x ** b * L ** c for x in xs], color="0.35", linestyle=(0, (3, 2)), linewidth=0.8)
ax.plot([], [], color="0.35", linestyle=(0, (3, 2)), linewidth=0.8, label="Power-law fit")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(chips)
ax.set_xticklabels([str(n) for n in chips])
ax.set_xlim(0.85, 9.4)
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]))
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylim(0.045, 30)
ax.set_xlabel("Chips")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=2.2, borderaxespad=0.2)
fig.subplots_adjust(left=0.11, right=0.63, bottom=0.17, top=0.97)
ylabel_top(fig, ax, "Throughput (queries/s)")
smallest = save(fig, V2 / "figures/fig_scaling.pdf")
print(f"fig_scaling: smallest text {smallest} pt, width {fig.get_size_inches()[0]:.2f} in")
