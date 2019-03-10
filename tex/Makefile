.PHONY: run

run:
	pdflatex exemplo.tex && bibtex exemplo && pdflatex exemplo.tex && tput reset

.PHONY: clean

clean:
	rm -f *.aux *.lof *.log *.lot *.out *.toc *.pdf
