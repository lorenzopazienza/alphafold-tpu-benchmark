"""Automatic style check of the printed text.

    python3 paper-v3/tools/check_style.py

Reads sections/*.tex, captions.tex and the generated table notes, and prints every case it finds.
Hard failures (exit 1): an em dash or "---"; a filler word (thus, therefore, hence, moreover, furthermore, notably, and
similar words that add emphasis but no content); a ratio written as "times rarer" or "times less"; a British spelling;
and a variant of a term that GLOSSARY.md forbids. Words that the appendix quotes from version 1 are exempt from the
glossary check.
Listed for reading by hand, without failing: colons, semicolons, sentences with more than three commas, a possible
missing Oxford comma, and sentences of more than 35 words.
"""
import pathlib
import re
import sys

V2 = pathlib.Path(__file__).resolve().parents[1]
FILES = (sorted((V2 / "sections").glob("*.tex")) + [V2 / "captions.tex"]
         + sorted((V2 / "tables").glob("*_note.tex")))

FILLER = [r"\bthus\b", r"\btherefore\b", r"\bhence\b", r"\bmoreover\b", r"\bfurthermore\b", r"\bnotably\b",
          r"\bimportantly\b", r"\bcrucially\b", r"\bindeed\b", r"it is worth noting", r"\bin order to\b",
          r"\bleverag", r"\brobust", r"\bdelve", r"\bunderscore", r"\bhighlights?\b", r"\bhighlighted\b",
          r"\bsignifican", r"\bsurprising", r"\binteresting", r"\bclearly\b", r"\bnovel\b", r"\bdramatic",
          r"\bremarkabl", r"\bseamless", r"\butiliz", r"\bquite\b", r"\brather\b", r"\bsomewhat\b", r"\barguabl"]
RATIO = [r"times rarer", r"times less\b", r"times smaller", r"times fewer"]
BRITISH = [r"behaviour", r"colour", r"analys(?:e|ed|ing)\b", r"\w+isation\b", r"\w+ise[ds]?\b(?<!\bwise)(?<!otherwise)",
           r"modelling", r"labelled", r"\bcentre", r"favour", r"programme", r"\bwhilst\b", r"\bamongst\b",
           r"travelled", r"\bdefence\b", r"\blicence\b", r"catalogue", r"\bgrey\b"]
BRITISH_OK = {"otherwise", "precise", "exercise", "concise", "noise", "promise", "rise", "raise", "wise", "advise",
              "comprise", "comprises", "comprised", "surprise", "enterprise", "devise", "revise", "revised",
              "premise", "expertise", "likewise", "franchise", "supervise", "televised", "praise", "cruise",
              "paradise", "despise", "disguise", "arise", "arises", "arised", "raised", "raises", "rises",
              "promised", "exercised", "noises", "precisely"}


def glossary_patterns():
    text = (V2 / "GLOSSARY.md").read_text()
    block = text.split("```")[1]
    return [l.strip() for l in block.splitlines() if l.strip()]


def printed(tex):
    """Rough printed text: comments removed, commands that print nothing removed, arguments kept."""
    out = []
    for line in tex.splitlines():
        m = re.search(r"(?<!\\)%", line)
        out.append(line[:m.start()] if m else line)
    t = "\n".join(out)
    t = re.sub(r"\\begin\{(?:table\*?|figure\*?|itemize|tabular)\}(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", t)
    t = re.sub(r"\\(?:end|label|ref|cite|url|includegraphics|gentable|gennote|input|setlength|caption)\s*(?:\[[^\]]*\])?\{[^{}]*\}(?:\{[^{}]*\})?", "X", t)
    t = re.sub(r"\\(?:section|subsection)\*?\{([^{}]*)\}", r"\n\n\1.\n\n", t)
    t = re.sub(r"\\(?:texttt|emph|textsuperscript|code)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\item\b", "\n\n", t)
    t = t.replace("~", " ").replace("\\_", "_").replace("\\%", "%").replace("\\$", "$").replace("\\,", " ")
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)
    t = re.sub(r"[{}]", "", t)
    return t


def sentences(t):
    for para in re.split(r"\n\s*\n", t):
        p = " ".join(para.split())
        if not p:
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\$])", p):
            s = s.strip()
            if s:
                yield s


def main():
    hard, soft = [], []
    gloss = glossary_patterns()
    n_sent, n_words = 0, 0
    for f in FILES:
        raw = f.read_text()
        for i, line in enumerate(raw.splitlines(), 1):
            body = re.split(r"(?<!\\)%", line)[0]
            if "---" in body or "\u2014" in body:
                hard.append(f"{f.name}:{i}: em dash or ---")
        t = printed(raw)
        # words quoted from version 1 (\vonequote) are exempt from the glossary check. Each quote
        # is removed from the sentence before the check, so a period inside a quote does not shift the sentences.
        vq = [" ".join(printed(q).split()) for q in re.findall(r"\\vonequote\{([^{}]*)\}", raw)]

        def unquoted(s):
            for q in vq:
                s = s.replace(q, "vonequote")
            return s
        for s in sentences(t):
            n_sent += 1
            w = len(s.split())
            n_words += w
            low = s.lower()
            where = f"{f.name}: {s[:110]}"
            for pat in FILLER:
                if re.search(pat, low):
                    hard.append(f"[filler {pat}] {where}")
            for pat in RATIO:
                if re.search(pat, low):
                    hard.append(f"[ratio form {pat}] {where}")
            for pat in BRITISH:
                for m in re.finditer(pat, low):
                    if m.group(0) not in BRITISH_OK:
                        hard.append(f"[British spelling {m.group(0)}] {where}")
            for pat in gloss:
                if re.search(pat, unquoted(s), re.IGNORECASE):
                    hard.append(f"[glossary variant {pat}] {where}")
            if re.search(r":\s", s):
                soft.append(f"[colon, read by hand] {f.name}: {s}")
            if ";" in s:
                soft.append(f"[semicolon] {f.name}: {s}")
            commas = len(re.findall(r",", re.sub(r"\[[^\]]*\]|\d,\d", "", s)))
            if commas > 3:
                soft.append(f"[{commas} commas] {f.name}: {s}")
            for m in re.finditer(r"(?:^|, )([^,;:.]{1,40}), ([^,;:.]{1,40}) (and|or) ", s):
                pass
            for m in re.finditer(r"\b[\w$=.-]+, [\w$=. -]{1,30}?[^,] (?:and|or) [\w$.-]+", s):
                seg = m.group(0)
                if seg.count(",") == 1 and not re.search(r"\b(?:between|from)\b", seg):
                    soft.append(f"[Oxford comma? '{seg}'] {f.name}: {s[:120]}")
            if w > 35:
                soft.append(f"[{w} words] {f.name}: {s}")
    print(f"sentences: {n_sent}, mean words: {n_words / max(n_sent, 1):.1f}")
    print(f"hard failures: {len(hard)}")
    for h in hard:
        print("  " + h)
    print(f"cases to read by hand: {len(soft)}")
    for s in soft:
        print("  " + s)
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
