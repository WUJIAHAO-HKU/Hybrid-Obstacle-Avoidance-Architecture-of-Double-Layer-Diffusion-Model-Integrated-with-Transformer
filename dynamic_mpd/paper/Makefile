# Makefile for IROS Paper Compilation

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

# Output directory
OUT = build

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf &

quick:
	$(LATEX) $(MAIN).tex

# Word count (approximate)
count:
	@echo "Approximate word count:"
	@pdftotext $(MAIN).pdf - 2>/dev/null | wc -w

help:
	@echo "Available targets:"
	@echo "  all   - Full compilation with bibliography"
	@echo "  quick - Single pdflatex run"
	@echo "  clean - Remove auxiliary files"
	@echo "  view  - Open PDF viewer"
	@echo "  count - Word count"
