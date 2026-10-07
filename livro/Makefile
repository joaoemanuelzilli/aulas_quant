# Compila o livro inteiro:  make        Limpa auxiliares:  make limpar
# Refaz todas as figuras:   make figuras
PY = ../.venv/bin/python

livro.pdf: livro.tex lmfquant.sty referencias.tex capitulos/*.tex
	lualatex -interaction=nonstopmode -halt-on-error livro.tex
	lualatex -interaction=nonstopmode -halt-on-error livro.tex
	lualatex -interaction=nonstopmode -halt-on-error livro.tex

figuras:
	scripts/refazer_figuras.sh

limpar:
	rm -f livro.aux livro.log livro.toc livro.out livro.idx livro.ind livro.ilg
	rm -rf build/cap* build/frente build/ap_* build/ft build/bis

.PHONY: figuras limpar
