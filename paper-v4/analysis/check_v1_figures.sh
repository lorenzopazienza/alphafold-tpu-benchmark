#!/usr/bin/env bash
# Regenerate the five v1 figures from the committed data with the v1 plotting scripts, in a scratch folder,
# and compare their text (labels and numbers) with the figures in paper/preprint/source/figures/.
# Needs python3 with matplotlib (e.g. a venv) and poppler's pdftotext. Writes nothing inside the repository.
# Usage: bash paper-v4/analysis/check_v1_figures.sh /path/to/python-with-matplotlib
set -euo pipefail
PY=${1:-python3}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(mktemp -d)
mkdir -p "$WORK/tools" "$WORK/out"
cp "$ROOT/scripts/plot_paper_figures.py" "$ROOT/scripts/plot_speedup_hardware.py" "$WORK/tools/"
(cd "$WORK" && "$PY" -I tools/plot_paper_figures.py --repo "$ROOT" --out "$WORK/out" >/dev/null)
(cd "$WORK/tools" && "$PY" -I plot_speedup_hardware.py >/dev/null && cp "$WORK/figures/speedup_hardware.pdf" "$WORK/out/")
for f in "$WORK"/out/*.pdf; do
  n=$(basename "$f")
  python3 - "$ROOT/paper/preprint/source/figures/$n" "$f" "$n" <<'PYEOF'
import collections, subprocess, sys
tok = lambda p: collections.Counter(subprocess.run(["pdftotext", p, "-"], capture_output=True, text=True).stdout.split())
a, b = tok(sys.argv[1]), tok(sys.argv[2])
print(sys.argv[3], "same text tokens" if a == b else f"differs: only v1 {dict(a-b)}, only new {dict(b-a)}")
PYEOF
done
echo "scratch folder: $WORK"
