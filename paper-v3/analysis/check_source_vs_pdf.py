"""Check that the v1 LaTeX source matches the v1 PDF.

Extracts every decimal number from the .tex sources (in \\input order) and
from the PDF text (pdftotext), and reports numbers present in one but not
the other. Also checks a set of sentinel phrases.
Usage: python3 paper-v3/analysis/check_source_vs_pdf.py
"""
import re, subprocess, collections, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "paper/preprint/source"
PDF = ROOT / "paper/preprint/Pazienza_ElBani_2026_AlphaFold2_Inference_on_Cloud_TPUs.pdf"

main = (SRC / "main.tex").read_text()
def expand(t):
    return re.sub(r"\\input\{([^}]*)\}", lambda m: expand((SRC / (m.group(1) + ".tex")).read_text()), t)
tex = expand(main)
tex = re.sub(r"(?<!\\)%.*", "", tex)            # drop comments
pdf = subprocess.run(["pdftotext", str(PDF), "-"], capture_output=True, text=True).stdout
pdf = pdf.replace("\u2212", "-").replace("ﬁ", "fi").replace("ﬂ", "fl")
num = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*")
def nums(t):
    t = t.replace("{,}", ",")
    return collections.Counter(n.replace(",", "") for n in num.findall(t) if "." in n or len(n) > 1)
a, b = nums(tex), nums(pdf)
only_tex = sorted(set(a) - set(b)); only_pdf = sorted(set(b) - set(a))
print("distinct numbers in tex:", len(a), " in pdf:", len(b))
print("in tex but not in pdf:", only_tex)
print("in pdf but not in tex:", only_pdf[:80])
tex_ws = re.sub(r"\s+", " ", tex.replace("\\\\", " ")); pdf_ws = re.sub(r"\s+", " ", pdf)
for phrase in ["Accelerator Choice Is Not Enough", "451", "27.8", "6.92", "0.981", "0.538", "463"]:
    squash = lambda x: re.sub(r"[^a-z0-9.]", "", x.lower())  # small caps split words in the pdf text
    print(f"{phrase!r}: tex={squash(phrase) in squash(tex_ws)} pdf={squash(phrase) in squash(pdf_ws)}")
print("note: 0.17 in the tex is a column width (p{0.17\\linewidth}); numbers only in the pdf come from figures, bibliography and page numbers")
