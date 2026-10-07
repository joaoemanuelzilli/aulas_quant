#!/usr/bin/env bash
# Regenera todas as figuras do livro (roda cada scripts/capNN/fig_*.py).
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; PY="$RAIZ/../.venv/bin/python"
export OMP_NUM_THREADS=4
cd "$RAIZ/scripts" && "$PY" fig_capa.py
for d in "$RAIZ"/scripts/cap*/; do
  cd "$d"
  for f in fig_*.py; do
    [ -e "$f" ] || continue
    "$PY" "$f" > /dev/null 2> "/tmp/fig_err_$$.txt" || { echo "FALHOU: $d$f"; tail -3 "/tmp/fig_err_$$.txt"; }
  done
done
rm -f "/tmp/fig_err_$$.txt"; echo "figuras refeitas"
