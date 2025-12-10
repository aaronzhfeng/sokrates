# Makefile for SOKRATES paper

PAPER = sokrates
LATEX = pdflatex
BIBTEX = bibtex

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf 2>/dev/null || open $(PAPER).pdf

.PHONY: all clean distclean view

