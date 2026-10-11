"""Confere o Capítulo 1 contra a execução dos scripts.

Rode de dentro de livro/scripts/cap01:  python conf_saidas.py
1. cada bloco style=saida de capitulos/cap01.tex é trecho literal da saída
   do script impresso logo antes dele;
2. nenhuma linha de código impresso passa de 66 caracteres;
3. os asserts dos próprios scripts passam (um script que falha derruba o teste).
"""
import re
import subprocess
import sys
from pathlib import Path

TEX = Path("../../capitulos/cap01.tex").read_text(encoding="utf8")
PADRAO = re.compile(
    r"\\lstinputlisting\[[^\]]*\]\{scripts/cap01/(\w+\.py)\}\s*"
    r"\\begin\{lstlisting\}\[style=saida\]\n(.*?)\\end\{lstlisting\}", re.S)


def sem_espaco_final(texto):
    return "\n".join(linha.rstrip() for linha in texto.strip("\n").splitlines())


blocos = PADRAO.findall(TEX)
assert len(blocos) == 4, f"esperava 4 pares código/saída, achei {len(blocos)}"
for script, colado in blocos:
    longas = [l for l in Path(script).read_text(encoding="utf8").splitlines() if len(l) > 66]
    assert not longas, f"{script}: linha com mais de 66 caracteres: {longas[0]}"
    r = subprocess.run([sys.executable, "-W", "error", script], capture_output=True, text=True)
    assert r.returncode == 0, f"{script} falhou:\n{r.stderr}"
    assert sem_espaco_final(colado) in sem_espaco_final(r.stdout), f"{script}: saída colada difere da execução"
    print(f"ok  {script}: {len(colado.strip().splitlines())} linhas de saída conferidas")
print("todas as saídas do capítulo conferem")
