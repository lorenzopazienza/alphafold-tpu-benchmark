"""Check of the build and of the printed sizes.

    .venv/bin/python tools/check_pdf.py Pazienza_ElBani_AlphaFold2_TPU_v3.pdf

- build: "??" in the text, undefined references or citations, overfull boxes (from build/main.log);
- pages, and words of the text (abstract to the end of the Acknowledgment, without tables, figure, and references);
- the smallest printed font size inside each table and figure, read from the PDF (PyMuPDF spans): the float region is
  found from its caption, and every span inside it is measured. Superscript markers are listed apart;
- a grayscale rendering of every page (build/gray/), to look at.
"""
import pathlib
import re
import subprocess
import sys

import pymupdf as fitz  # PyMuPDF

V2 = pathlib.Path(__file__).resolve().parents[1]
pdf = V2 / (sys.argv[1] if len(sys.argv) > 1 else "Pazienza_ElBani_AlphaFold2_TPU_v3.pdf")
# the log of the build that made this PDF: build/main.log, or build/main_noappendix.log for the version without the
# appendix
NOAPP = "noappendix" in pdf.name
log = (V2 / ("build/main_noappendix.log" if NOAPP else "build/main.log")).read_text(errors="replace")
doc = fitz.open(pdf)
fails = []

text = "".join(p.get_text() for p in doc)
print(f"pages: {doc.page_count}")
if "??" in text:
    fails.append("?? in the PDF text")
for pat in [r"Citation .* undefined", r"Reference .* undefined", r"There were undefined", r"Overfull \\hbox", r"Overfull \\vbox"]:
    hits = re.findall(pat + r"[^\n]*", log)
    print(f"log {pat!r}: {len(hits)}")
    fails += hits

# words of the text: from "Abstract" to "References", minus the floats (their spans are removed below)
floats = []
for pno, page in enumerate(doc):
    d = page.get_text("dict")
    for b in d["blocks"]:
        for l in b.get("lines", []):
            s = "".join(sp["text"] for sp in l["spans"]).strip()
            if re.match(r"^(TABLE [IVX]+|Fig\. \d+\.)", s):
                floats.append((pno, s.split(".")[0] if s.startswith("Fig") else s, l["bbox"]))

def region(pno, label, bbox):
    """The rectangle of a float: a table from its label down to the end of its note; the figure from the top of the
    page (or column) down to its caption."""
    page = doc[pno]
    W = page.rect.width
    x0, y0, x1, y1 = bbox
    full = abs((x0 + x1) / 2 - W / 2) < 20          # a full-width float has its label centred on the page
    if label.startswith("TABLE"):
        # down to the last line in small type below the label, before the body text resumes (10 pt)
        ys = y1
        lines = sorted((l for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])), key=lambda l: l["bbox"][1])
        for l in lines:
            if True:
                bx0, by0, bx1, by1 = l["bbox"]
                sizes = [sp["size"] for sp in l["spans"] if sp["text"].strip()]
                if not sizes or by0 < y0:
                    continue
                same_col = full or (bx0 >= x0 - 120 and bx1 <= x1 + 120)
                if same_col and max(sizes) < 9.6 and by0 - ys < 14:
                    ys = max(ys, by1)
        return fitz.Rect(0 if full else (0 if x0 < W / 2 else W / 2), y0, W if full else (W / 2 if x0 < W / 2 else W), ys)
    # figure: the image above its caption, in the caption's column
    col = (0, W / 2) if x0 < W / 2 else (W / 2, W)
    top = 0
    return fitz.Rect(col[0], top, col[1], y0 - 1)

print("smallest printed text in each float (pt):")
for pno, label, bbox in floats:
    r = region(pno, label, bbox)
    sizes, sup, caption = [], [], []
    for b in doc[pno].get_text("dict", clip=r)["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]:
                if not sp["text"].strip() or not r.intersects(fitz.Rect(sp["bbox"])):
                    continue
                if label.startswith("Fig") and "Arial" not in sp["font"]:
                    continue          # the figure is the only text set in Arial
                if label.startswith("TABLE") and sp["bbox"][1] < bbox[3] + 11:
                    caption.append(round(sp["size"], 2))
                    continue          # the caption title (IEEEtran small capitals), reported apart
                if sp["size"] < 7.9 and len(sp["text"].strip()) <= 1:
                    sup.append((sp["text"].strip(), round(sp["size"], 2)))
                else:
                    sizes.append((round(sp["size"], 2), sp["text"].strip()[:20]))
    m = min(sizes) if sizes else None
    print(f"  page {pno + 1}, {label}: smallest {m[0] if m else '-'} pt ({m[1] if m else ''!r}); superscript markers: {sup}"
          + (f"; caption title {min(caption)}-{max(caption)} pt (IEEEtran small capitals)" if caption else ""))
    if m and m[0] < 7.95:
        fails.append(f"{label}: text at {m[0]} pt")

out = V2 / ("build/gray_noappendix" if NOAPP else "build/gray")
out.mkdir(parents=True, exist_ok=True)
subprocess.run(["pdftoppm", "-gray", "-r", "80", "-png", str(pdf), str(out / "p")], check=True)
print(f"grayscale pages in {out}")

# every plain sentence of the source must be in the PDF (a "% src" comment once swallowed the rest of its line)
sys.path.insert(0, str(V2 / "tools"))
from check_style import printed, sentences
norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
pdf_norm = norm(re.sub(r"-\n", "", text))
missing, n_sent = [], 0


def found_split(ns, k=3):
    """ns is in the PDF text, possibly cut into up to k + 1 pieces by floats or footnotes."""
    if ns in pdf_norm:
        return True
    if k == 0:
        return False
    lo, hi = 0, len(ns)
    while lo < hi:                       # longest prefix present in the PDF text
        mid = (lo + hi + 1) // 2
        if ns[:mid] in pdf_norm:
            lo = mid
        else:
            hi = mid - 1
    # a float can cut a sentence after a short first word ("Ten"): accept a short prefix if a long rest follows
    return (lo >= 8 or (lo >= 3 and len(ns) - lo >= 20)) and found_split(ns[lo:], k - 1)
has_app = "Changes from Earlier Versions" in text or "CHANGES FROM EARLIER VERSIONS" in text.upper()
print(f"appendix in the PDF: {has_app} (expected {not NOAPP})")
if has_app == NOAPP:
    fails.append("appendix presence does not match the build")
for f in sorted((V2 / "sections").glob("*.tex")):
    if f.name == "appendix.tex" and not has_app:
        continue
    src_tex = f.read_text()
    if not has_app:
        src_tex = re.sub(r"\\ifappendix.*?\\fi", "", src_tex, flags=re.S)
    for s in sentences(printed(src_tex)):
        if re.search(r"[$\\]|\bX\b|_", s) or len(s.split()) < 4:
            continue
        n_sent += 1
        ns = norm(s)
        if ns not in pdf_norm:
            # a float or a footnote can interrupt a sentence in the extracted text: both halves must then be present
            if not found_split(ns):
                missing.append(f"{f.name}: {s[:100]}")
print(f"source sentences found in the PDF: {n_sent - len(missing)} of {n_sent}")
fails += [f"not in the PDF: {m}" for m in missing]

# words of the text
body = text.split("References")[0] if "References" in text else text
body = body.split("Abstract")[1] if "Abstract" in body else body
print(f"words from the abstract to the references (pdftotext-like, floats included): {len(body.split())}")
if has_app:
    app = re.split(r"C\s*HANGES\s+FROM\s+E\s*ARLIER\s+V\s*ERSIONS|Changes from Earlier Versions", text, flags=re.I)[-1]
    print(f"words of the appendix (pdftotext-like): {len(app.split())}")
print(f"failures: {len(fails)}")
for f in fails:
    print("  " + f)
sys.exit(1 if fails else 0)
