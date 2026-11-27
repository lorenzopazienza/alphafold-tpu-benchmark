"""Check of the numbers: every number printed in the hand-written LaTeX has a "% src:" comment, and matches a value of the
source it names, at the printed number of decimals (round half up).

    python3 paper-v4/tools/check_numbers.py [--verbose]

Scope: sections/*.tex and captions.tex. The tables and their notes are generated from the data by make_tables.py and
are not hand-written. A block is the run of text lines up to and including the next line with a "% src:" comment;
every number in a block must match a value named by that comment. A paragraph (blank line) that ends with numbers
and no "% src:" is an error.

Source items, separated by ";" in the comment:
- recomputed: KEY, KEY (...)          values of paper-v4/analysis/out/recomputed.json
- <file>.json -> PATH, PATH           values at PATH in that file ([i], [*], a.b, len(x), keys); no PATH: the whole file
- <file> ("quote")                    the quote must occur in the file; numbers in the quote count as values
- anything else ("quote")             an external source (bibliography, web page): numbers in the quote count, and the
                                      item is listed as not machine-checked
A quote attached to a recomputed item must occur in recomputed.json.

Matching: a printed number p with d decimals matches a value v if round_half_up(|v|, d) == p, or if v is a byte count
and v / 10^6 rounds to p (MB). A truncation (floor of |v| at d decimals, for "at least" bounds) also matches and is
listed. Version strings (0.10.2) match the same string.
"""
import json
import pathlib
import re
import sys
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

V2 = pathlib.Path(__file__).resolve().parents[1]
ROOT = V2.parent
REC = json.loads((V2 / "analysis/out/recomputed.json").read_text())
REC_TEXT = (V2 / "analysis/out/recomputed.json").read_text()
FILES = sorted((V2 / "sections").glob("*.tex")) + [V2 / "captions.tex"]
VERBOSE = "--verbose" in sys.argv

STRIP_CMDS = r"\\(?:cite|ref|label|url|includegraphics|gentable|gennote|input|setlength|section\*?|subsection|begin|end|bstctlcite|caption)\s*(?:\[[^\]]*\])?\{[^{}]*\}(?:\{[^{}]*\})?"
NUM = re.compile(r"(?<![A-Za-z0-9_.\\])(\d+(?:\.\d+)+|\d+)(?![A-Za-z0-9_])")


def split_comment(line):
    m = re.search(r"(?<!\\)%", line)
    return (line, "") if not m else (line[:m.start()], line[m.start() + 1:])


def numbers_in_text(text):
    t = re.sub(STRIP_CMDS, " ", text)
    t = re.sub(r"\$?R\^\{?2\}?\$?", " ", t)          # R^2 is a name, not a value
    t = re.sub(r"\\_\d+", " ", t)                     # model\_3 is a name
    t = re.sub(r"arXiv:\d{4}\.\d{4,5}", " ", t)        # the arXiv identifier is a name
    t = re.sub(r"\b[Vv]ersions? \d(?: and \d)?\b", " ", t)  # "version 1", "versions 1 and 2" are names of the paper versions
    t = re.sub(r"\\looseness=-?\d+", " ", t)         # layout command, prints nothing
    return NUM.findall(t)


def flatten(x):
    out = []
    if isinstance(x, dict):
        for v in x.values():
            out += flatten(v)
    elif isinstance(x, list):
        for v in x:
            out += flatten(v)
    elif isinstance(x, bool):
        pass
    elif isinstance(x, (int, float)):
        out.append(x)
    elif isinstance(x, str):
        out += numbers_from_string(x)
    return out


def numbers_from_string(s):
    out = []
    for tok in re.findall(r"\d+(?:\.\d+)+|\d+", s):
        out.append(tok if tok.count(".") > 1 else (float(tok) if "." in tok else int(tok)))
    return out


def walk(obj, path):
    """Values at a path like [1].memory_stats_per_device[*].bytes_in_use, grid[*].length, len(grid), keys."""
    path = path.strip()
    if path == "keys":
        return list(obj.keys())
    m = re.fullmatch(r"len\((.*)\)", path)
    if m:
        return [len(x) for x in walk_list(obj, m.group(1))]
    return walk_list(obj, path)


def walk_list(obj, path):
    cur = [obj]
    for part in re.findall(r"\[[^\]]*\]|[^.\[\]]+", path):
        nxt = []
        for c in cur:
            if part == "[*]":
                nxt += list(c)
            elif part.startswith("["):
                nxt.append(c[int(part[1:-1])])
            else:
                if part not in c:
                    raise KeyError(path)
                nxt.append(c[part])
        cur = nxt
    return cur


def resolve(src):
    """Values named by a src comment. Returns (values, problems, external quotes)."""
    vals, problems, external = [], [], []
    for item in [i.strip() for i in src.split(";") if i.strip()]:
        quotes = re.findall(r'"([^"]+)"', item)
        bare = re.sub(r'\([^()]*"[^"]*"[^()]*\)', "", item)
        bare = re.sub(r'"[^"]*"', "", bare).strip()
        bare = re.sub(r"\s+\([^()\"]*\)", "", bare).strip()     # row codes such as (H-03, I-03) are not paths or keys
        if bare.startswith("recomputed:"):
            for k in [k.strip().split()[0] for k in bare[len("recomputed:"):].split(",") if k.strip()]:
                if k in REC["values"]:
                    vals += flatten(REC["values"][k])
                else:
                    problems.append(f"unknown recomputed key {k}")
            for q in quotes:
                if q not in REC_TEXT:
                    problems.append(f"quote not in recomputed.json: {q!r}")
                vals += numbers_from_string(q)
            continue
        m = re.match(r"(\S+\.json)\s*(?:->\s*(.*))?$", bare)
        if m and (ROOT / m.group(1)).exists():
            data = json.loads((ROOT / m.group(1)).read_text())
            paths = [p.strip() for p in (m.group(2) or "").split(",") if p.strip()]
            if not paths:
                vals += flatten(data)
            for p in paths:
                try:
                    vals += flatten(walk(data, p.split()[0]))
                except (KeyError, IndexError, TypeError, ValueError):
                    problems.append(f"path not found: {m.group(1)} -> {p}")
            continue
        m = re.match(r"(\S+\.(?:yaml|py|md|txt|tex|json))", bare)
        if m and (ROOT / m.group(1)).exists():
            text = (ROOT / m.group(1)).read_text()
            for q in quotes:
                if " ".join(q.split()) not in " ".join(text.split()):
                    problems.append(f"quote not in {m.group(1)}: {q!r}")
                vals += numbers_from_string(q)
            continue
        if m:
            problems.append(f"file not found: {m.group(1)}")
            continue
        for q in quotes:
            vals += numbers_from_string(q)
        external.append(item[:90])
    return vals, problems, external


def matches(p, vals):
    """'' if no match, 'round' or 'truncation' otherwise."""
    if p.count(".") > 1:
        return "round" if p in [v for v in vals if isinstance(v, str)] else ""
    d = len(p.split(".")[1]) if "." in p else 0
    target = Decimal(p)
    q = Decimal(1).scaleb(-d)
    nums = [v for v in vals if not isinstance(v, str)]
    for v in nums:
        a = abs(Decimal(repr(v)))
        if a.quantize(q, rounding=ROUND_HALF_UP) == target:
            return "round"
        if a >= 10 ** 5 and (a / Decimal(10 ** 6)).quantize(q, rounding=ROUND_HALF_UP) == target:
            return "round"
    for v in nums:
        a = abs(Decimal(repr(v)))
        if a.quantize(q, rounding=ROUND_DOWN) == target:
            return "truncation"
    return ""


def main():
    n_checked, errors, truncs, externals = 0, [], [], []
    for f in FILES:
        block = []
        for i, line in enumerate(f.read_text().splitlines(), 1):
            text, comment = split_comment(line)
            if text.strip() == "" and comment.strip() == "":
                if block:
                    errors.append(f"{f.name}:{block[0][0]}: no % src for {[n for _, n in block]}")
                block = []
                continue
            block += [(i, n) for n in numbers_in_text(text)]
            c = comment.strip()
            if c.startswith("src:"):
                vals, problems, external = resolve(c[4:])
                errors += [f"{f.name}:{i}: {p}" for p in problems]
                externals += [f"{f.name}:{i}: {e}" for e in external]
                for ln, n in block:
                    n_checked += 1
                    how = matches(n, vals)
                    if not how:
                        errors.append(f"{f.name}:{ln}: {n} not found in its source ({c[4:80]}...)")
                    elif how == "truncation":
                        truncs.append(f"{f.name}:{ln}: {n} (truncated, an 'at least' bound)")
                    elif VERBOSE:
                        print(f"ok {f.name}:{ln}: {n}")
                block = []
        if block:
            errors.append(f"{f.name}:{block[0][0]}: no % src for {[n for _, n in block]}")
    print(f"numbers checked: {n_checked}")
    print(f"truncations (at-least bounds): {len(truncs)}")
    for t in truncs:
        print("  " + t)
    print(f"external sources, quotes not machine-checked: {len(externals)}")
    for e in externals:
        print("  " + e)
    print(f"errors: {len(errors)}")
    for e in errors:
        print("  " + e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
