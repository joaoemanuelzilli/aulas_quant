# Compila o livro inteiro:                    make
# Refaz os dados derivados e as figuras:      make figuras
# Refaz e valida os arquivos de dados/:       make dados
# Limpa auxiliares e compilações isoladas:    make limpar
# O Python usado por "make figuras" e "make dados" é o do ambiente virtual em ../.venv;
# para trocar:  make figuras PY=/caminho/do/python
PY ?= $(abspath ../.venv/bin/python)
LATEX = lualatex -interaction=nonstopmode -halt-on-error

# Tudo o que entra no PDF: texto, estilo, figuras e os códigos impressos.
FONTES = livro.tex lmfquant.sty lmfquant.lua referencias.tex hyph-pt.pat.txt \
         $(wildcard capitulos/*.tex) $(wildcard figuras/*.pdf figuras/*/*.pdf) \
         $(wildcard marca/*) $(wildcard scripts/cap*/*.py)

livro.pdf: $(FONTES)
	$(LATEX) livro.tex
	$(LATEX) livro.tex
	$(LATEX) livro.tex

figuras:
	PY="$(PY)" scripts/refazer_figuras.sh

# Regrava dados/ a partir de dados/bruto/ (extrato oficial da B3, tabelas de
# eventos e de correções, cópias do Yahoo) e roda as validações, que terminam
# com erro se alguma falhar. O extrato dados/bruto/b3_cotahist.csv é feito uma
# única vez por scripts/extrair_cotahist.py, que depende da rede.
dados:
	"$(PY)" scripts/limpar_dados.py
	"$(PY)" scripts/validar_dados.py

# Apaga só o que é regenerável. build/revisao (relatórios das revisões)
# e as demais pastas de build/ ficam.
limpar:
	rm -f livro.aux livro.log livro.toc livro.out livro.idx livro.ind livro.ilg
	rm -rf build/cap[0-9][0-9]* build/frente build/ap_a_notacao build/ap_b_glossario build/ft build/bis

.PHONY: figuras dados limpar
