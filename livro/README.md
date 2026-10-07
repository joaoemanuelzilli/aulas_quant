# Finanças quantitativas com Python

Livro didático do Núcleo de Análise Quantitativa da Liga de Mercado Financeiro (UFU).
Autor: João Emanuel Zilli dos Santos. O PDF pronto é `livro.pdf`.

## O que há nesta pasta

| Caminho | Conteúdo |
|---|---|
| `livro.tex` | arquivo principal |
| `capitulos/` | um `.tex` por capítulo, mais capa, prefácio e apêndices |
| `referencias.tex` | bibliografia (cada capítulo cita só chaves definidas aqui) |
| `lmfquant.sty`, `lmfquant.lua` | projeto gráfico: cores da Liga, fontes, caixas, código |
| `scripts/capNN/` | códigos impressos no livro (`l*.py`) e geradores de figuras (`fig_*.py`) |
| `scripts/estilo.py` | estilo único das figuras |
| `figuras/capNN/` | figuras em PDF, geradas pelos scripts |
| `dados/bruto/` | cópia exata do que foi baixado (Yahoo Finance, outubro de 2026) |
| `dados/` | arquivos limpos por `scripts/limpar_dados.py`, lidos pelos capítulos |
| `fonts/`, `marca/` | fontes livres (STIX Two, Inter, JetBrains Mono) e logotipos |
| `GUIA_DE_ESTILO.md` | regras de escrita, rigor e figuras para quem for alterar o livro |

## Como compilar

Requisitos: LuaLaTeX (TeX Live 2023 ou mais novo, com `luaotfload`, `fontspec`, `unicode-math`,
`tcolorbox`, `tikz`, `listings`, `titlesec`, `imakeidx`) e Python 3.11+ com `requirements.txt`.

```
make                              # compila livro.pdf (três passadas)
scripts/compilar_capitulo.sh 05   # compila só um capítulo, em build/cap05/
make figuras                      # refaz todas as figuras
```

No Overleaf: envie a pasta inteira e escolha LuaLaTeX como compilador.

## Como alterar

1. Leia `GUIA_DE_ESTILO.md` e use `capitulos/cap01.tex` como modelo.
2. Todo número do texto sai de um script em `scripts/capNN/`; toda saída impressa é colada da
   execução real. Ao mudar um script ou os dados, rode de novo e atualize o texto.
3. Não edite `dados/bruto/`. Para atualizar os dados, apague a pasta, rode
   `scripts/baixar_dados.py` e `scripts/limpar_dados.py`, e depois rode todos os scripts dos
   capítulos: os números do livro vão mudar e precisam ser atualizados no texto.
4. Compile o capítulo, olhe o PDF e só então compile o livro inteiro.

Material educacional. Nada aqui é recomendação de investimento.
