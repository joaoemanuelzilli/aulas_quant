#!/usr/bin/env bash
# Regenera os dados derivados e todas as figuras do livro.
#
#   scripts/refazer_figuras.sh            todos os capítulos
#   scripts/refazer_figuras.sh 06 08      só os capítulos indicados
#   PY=/caminho/do/python scripts/refazer_figuras.sh
#
# Em cada capítulo roda primeiro os GERADORES e depois os fig_*.py.
# Gerador é todo script de scripts/capNN/ que grava um arquivo em
# dados/ (os dados/cNN_*.csv que as figuras leem), exceto os
# baixar_*.py, que dependem da rede e são rodados uma única vez à mão.
# Cada script é executado de dentro da própria pasta.
# Termina com código 1 se algum script falhar.
set -u
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$RAIZ/../.venv/bin/python}"
case "$PY" in
  /*) ;;                                              # caminho absoluto
  */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")" ;;
  *) PY="$(command -v "$PY" || echo "$PY")" ;;        # nome no PATH
esac
[ -x "$PY" ] || { echo "Python não encontrado em '$PY' (defina PY=...)"; exit 1; }
export OMP_NUM_THREADS=4 MPLBACKEND=Agg

ERRO="$(mktemp)"; trap 'rm -f "$ERRO"' EXIT
rodados=0; falhas=0
rodar() {                       # rodar <pasta> <script>
  rodados=$((rodados + 1))
  ( cd "$1" && "$PY" "$2" > /dev/null 2> "$ERRO" ) && return 0
  echo "FALHOU: ${1#"$RAIZ"/}/$2"; tail -3 "$ERRO"; falhas=$((falhas + 1))
}
geradores() {                   # geradores <pasta>: um nome por linha
  grep -l -E 'to_csv|to_pickle|to_parquet|np\.savez?\(' "$1"/*.py 2> /dev/null |
    xargs -r -n1 basename | grep -v -E '^(baixar_|fig_)' | sort
}

if [ "$#" -gt 0 ]; then
  PASTAS=""
  for n in "$@"; do PASTAS="$PASTAS $RAIZ/scripts/cap$(printf '%02d' "$((10#$n))")"; done
else
  [ -e "$RAIZ/scripts/fig_capa.py" ] && rodar "$RAIZ/scripts" fig_capa.py
  PASTAS="$(echo "$RAIZ"/scripts/cap*/)"
fi
for d in $PASTAS; do
  d="${d%/}"
  [ -d "$d" ] || { echo "FALHOU: pasta ${d#"$RAIZ"/} não existe"; falhas=$((falhas + 1)); continue; }
  for g in $(geradores "$d"); do rodar "$d" "$g"; done
  for f in "$d"/fig_*.py; do
    [ -e "$f" ] && rodar "$d" "$(basename "$f")"
  done
done

if [ "$falhas" -eq 0 ]; then
  echo "figuras refeitas ($rodados scripts)"
else
  echo "$falhas de $rodados scripts falharam"; exit 1
fi
