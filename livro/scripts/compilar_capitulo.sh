#!/usr/bin/env bash
# Compila um único capítulo para conferência:  scripts/compilar_capitulo.sh 03
# (aceita também "3"). Pode ser chamado de qualquer pasta.
# Saída em build/cap03/cap03.pdf (não interfere na compilação do livro inteiro).
# Referências a outros capítulos aparecem como "??": é esperado.
set -u
[ "$#" -eq 1 ] || { echo "uso: scripts/compilar_capitulo.sh NN   (01 a 12)"; exit 1; }
N="$(printf '%02d' "$((10#$1))")"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
[ -e "capitulos/cap$N.tex" ] || { echo "capitulos/cap$N.tex não existe"; exit 1; }
mkdir -p "build/cap$N"
cat > "build/cap$N/cap$N.tex" <<TEX
\documentclass[10pt, oneside, openany]{book}
\usepackage{lmfquant}
\begin{document}
\mainmatter
\setcounter{chapter}{$((10#$N - 1))}
\input{capitulos/cap$N}
\backmatter
\input{referencias}
\end{document}
TEX
for i in 1 2; do
  lualatex -interaction=nonstopmode -halt-on-error -output-directory="build/cap$N" "build/cap$N/cap$N.tex" > "build/cap$N/saida.txt" 2>&1 || { grep -n -A12 '^!' "build/cap$N/saida.txt" | head -40; echo "ERRO na compilação"; exit 1; }
done
echo "ok: build/cap$N/cap$N.pdf  ($(pdfinfo "build/cap$N/cap$N.pdf" | awk '/Pages/{print $2}') páginas)"
grep -E "Overfull .hbox \(([5-9]|[1-9][0-9]+)\.[0-9]+pt" "build/cap$N/saida.txt" | sort | uniq -c | head -20
grep -E "Citation .* undefined|Missing character" "build/cap$N/saida.txt" | sort -u | head -20
