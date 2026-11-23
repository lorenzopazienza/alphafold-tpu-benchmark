# Build of version 3 of the paper. Run from paper-v3/:  make        (tables, figure, PDF with and without Appendix A)
#                                                     make check  (checks of the numbers, the style, and the PDF)
# PYTHON needs matplotlib and Arial for the figure, and pymupdf for make check
# (python3 -m venv .venv && .venv/bin/pip install matplotlib==3.10.7 numpy==2.5.5 pymupdf==2.1). Other versions of
# matplotlib can move the legend and the y label of the figure by a few pixels.
PYTHON ?= .venv/bin/python
PDF = Pazienza_ElBani_AlphaFold2_TPU_v3.pdf
PDF_NOAPP = Pazienza_ElBani_AlphaFold2_TPU_v3_noappendix.pdf
SECTIONS = $(wildcard sections/*.tex)
TABLES = tables/tab_platforms.tex tables/tab_experiments.tex tables/tab_multichip.tex tables/tab_sweeps.tex
DATA = $(wildcard ../results/*.json ../results/sweep/*.json ../results/repro/*/*.json) analysis/out/recomputed.json

all: $(PDF) $(PDF_NOAPP)

analysis/out/recomputed.json: analysis/recompute.py
	cd .. && python3 -W ignore paper-v3/analysis/recompute.py

$(TABLES) &: tools/make_tables.py tools/common.py $(DATA)
	python3 tools/make_tables.py

figures/fig_scaling.pdf: tools/make_fig_scaling.py tools/common.py ../results/sweep/scaling_law_data.json analysis/out/recomputed.json
	cd tools && ../$(PYTHON) make_fig_scaling.py

$(PDF): main.tex macros.tex captions.tex references.bib $(SECTIONS) $(TABLES) figures/fig_scaling.pdf
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex >/dev/null
	cd build && BIBINPUTS=.. BSTINPUTS=.. bibtex main >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex >/dev/null
	cp build/main.pdf $(PDF)
	@grep -E "LaTeX Warning|Overfull|Underfull|undefined" build/main.log || echo "no warnings"

# The same paper without Appendix A (main_noappendix.tex defines \noappendix and reads main.tex).
$(PDF_NOAPP): main_noappendix.tex main.tex macros.tex captions.tex references.bib $(SECTIONS) $(TABLES) figures/fig_scaling.pdf
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main_noappendix.tex >/dev/null
	cd build && BIBINPUTS=.. BSTINPUTS=.. bibtex main_noappendix >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main_noappendix.tex >/dev/null
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main_noappendix.tex >/dev/null
	cp build/main_noappendix.pdf $(PDF_NOAPP)
	@grep -E "LaTeX Warning|Overfull|Underfull|undefined" build/main_noappendix.log || echo "no warnings"

check: $(PDF) $(PDF_NOAPP)
	python3 tools/check_numbers.py
	python3 tools/check_style.py
	$(PYTHON) -W ignore tools/check_pdf.py $(PDF)
	$(PYTHON) -W ignore tools/check_pdf.py $(PDF_NOAPP)

clean:
	rm -rf build tables figures $(PDF) $(PDF_NOAPP)

.PHONY: all check clean
