# constants
CLASSPATH=..
PDFLATEX=pdflatex -halt-on-error
DVIPS=dvips -q
BIBTEX=bibtex
DEBUG=pdflatex
 
# name of project
FILENAME=monografia

# extensions to clean
RM_EXT=*.aux *.tex~ *.vrb *.blg *.brf *.bak \
       *.loa *.log *.out *.tdo *.toc *.bib~ \
       *.tex~ *.pbm tmp*.ps tmp tmp*.@@@    \
       *.nav *.snm "\#"*."\#" *.bbl

# sane targets
all: bib pdf


pdf: $(FILENAME).tex
	@echo "**Generating PDF**"
	@$(PDFLATEX) $(FILENAME) > /dev/null


bib:
	@echo "**Adjusting References**"
	@$(PDFLATEX) $(FILENAME) > /dev/null
	@$(BIBTEX) $(FILENAME) > /dev/null
	@$(BIBTEX) $(FILENAME) > /dev/null
	@$(PDFLATEX) $(FILENAME) > /dev/null


clean: ;
	@echo "**Cleaning Files**"
	@rm -f $(RM_EXT)


debug: ;
	@$(DEBUG) $(FILENAME)


# Tell versions [3.59,3.63) of GNU make to not export all variables.
# Otherwise a system limit (for SysV at least) may be exceeded.
.NOEXPORT:
