"""Shared helpers of the generators of tables and figures.

Every generator reads the committed data files of the repository (results/) and paper-v3/analysis/out/recomputed.json.
No number is written by hand in a generator: values come from those files, and the generator only formats them.
Figure rules: Arial, no text below 8 pt, no rotated axis label (the y label is written horizontally above the axis),
and series told apart by marker or dash so that they read in grayscale.
"""
import json
import pathlib

V2 = pathlib.Path(__file__).resolve().parents[1]          # paper-v3/
ROOT = V2.parent                                          # worktree root
RECOMPUTED = V2 / "analysis/out/recomputed.json"
FS = 8.0                    # base font size of the figures, printed at natural size (column width 3.5 in)
COLUMN_IN = 252 / 72.27     # IEEEtran conference \columnwidth (252 pt), so the figure prints at natural size
YLAB_PAD_PT = 3.0


def load(rel):
    return json.loads((ROOT / rel).read_text())


def rv(key):
    """A value of recomputed.json."""
    return json.loads(RECOMPUTED.read_text())["values"][key]


def rhu(x, d):
    """Round half up on the decimal string of x (0.4875 -> 0.488), the rule the text and the number check use."""
    from decimal import ROUND_HALF_UP, Decimal
    return Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def rec(x):
    """A value as the file records it, without rounding or padding."""
    return repr(x) if isinstance(x, float) else str(x)


def fmt(x, d):
    """Fixed decimals, round half up, real minus sign in LaTeX math."""
    s = f"{rhu(x, d):f}"
    return s.replace("-", "$-$")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def style():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    path = font_manager.findfont("Arial", fallback_to_default=False)
    if pathlib.Path(path).name.lower() != "arial.ttf":
        raise SystemExit(f"Arial not found (got {path})")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "Arial", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "legend.title_fontsize": FS,
        "pdf.fonttype": 42, "mathtext.default": "regular", "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "legend.frameon": False, "axes.unicode_minus": True, "svg.hashsalt": "af2v2"})
    return plt


def ylabel_top(fig, ax, label):
    """Write the y label horizontally above the axis, left-aligned with the y tick labels. The figure grows at the
    top by the room the label needs, so the axes keep their size."""
    fs = FS
    ax.set_ylabel("")
    W, H = fig.get_size_inches()
    add = (YLAB_PAD_PT + 1.3 * fs) / 72
    H2 = H + add
    for a in fig.axes:
        q = a.get_position()
        a.set_position([q.x0, q.y0 * H / H2, q.width, q.height * H / H2])
    fig.set_size_inches(W, H2)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    yb = ax.yaxis.get_tightbbox(r)
    x0, y0 = yb.x0, max(ax.bbox.y1, yb.y1) + YLAB_PAD_PT * fig.dpi / 72
    t = fig.text(x0 / fig.bbox.width, y0 / fig.bbox.height, label, ha="left", va="bottom", fontsize=fs)
    fig._top_ylabel = t
    return t


def save(fig, out_pdf):
    """Checks before saving: no text below 8 pt, no rotated axis label, no text outside the page, the y label above
    the axis meets nothing else. Stops on any failure."""
    from matplotlib.text import Text
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    small = [(t.get_text(), t.get_fontsize()) for t in texts if t.get_fontsize() < 8]
    if small:
        raise SystemExit(f"{out_pdf.name}: text below 8 pt: {small[:3]}")
    for ax in fig.axes:
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_visible() and lab.get_text().strip() and round(lab.get_rotation()) % 360:
                raise SystemExit(f"{out_pdf.name}: rotated axis label {lab.get_text()!r}")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    for t in texts:
        bb = t.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            raise SystemExit(f"{out_pdf.name}: text outside the figure: {t.get_text()!r}")
    top = getattr(fig, "_top_ylabel", None)
    if top is not None:
        bb = top.get_window_extent(r)
        others = [o.get_window_extent(r) for o in texts if o is not top] + [a.bbox for a in fig.axes]
        if any(bb.overlaps(o) for o in others):
            raise SystemExit(f"{out_pdf.name}: the y label above the axis meets another element")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, metadata={"Software": None})
    return min(t.get_fontsize() for t in texts)
