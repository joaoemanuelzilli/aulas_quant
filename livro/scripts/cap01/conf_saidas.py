"""Confere o Capítulo 1 contra a execução dos scripts.

Rode de dentro de livro/scripts/cap01:  python conf_saidas.py
1. cada bloco style=saida de capitulos/cap01.tex é trecho literal da saída
   do script impresso logo antes dele;
2. nenhuma linha de código impresso passa de 66 caracteres;
3. os asserts dos próprios scripts passam (um script que falha derruba o teste).

Exceção: l06_custo.py mede tempos, que mudam a cada execução e de uma máquina
para outra. A saída impressa no livro é a execução gravada em l06_custo_saida.txt
(python l06_custo.py > l06_custo_saida.txt); aqui se confere que o texto colado é
essa gravação e que uma execução nova mantém as proporções que o texto afirma.
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


def confere_tempos(saida, nome):
    """Proporções afirmadas no texto para a saída de l06_custo.py."""
    linhas = saida.strip().splitlines()
    tab = [list(map(float, l.split())) for l in linhas[1:5]]
    lista = [t[1] for t in tab]
    conj = [t[2] for t in tab]
    assert [t[0] for t in tab] == [1e3, 1e4, 1e5, 1e6]
    for a, b in zip(lista, lista[1:]):          # lista: ~10x a cada 10x
        assert 5 < b / a < 20, f"{nome}: lista não cresce ~10x ({a}, {b})"
    assert max(conj) / min(conj) < 3, f"{nome}: conjunto não ficou constante"
    assert lista[-1] / conj[-1] > 1e5, f"{nome}: razão lista/conjunto < 100 mil"
    vezes = float(re.search(r"\((\d+) vezes\)", linhas[-1]).group(1))
    assert 30 <= vezes <= 100, f"{nome}: laço/NumPy = {vezes}, texto diz ~50"
    print(f"    {nome}: lista/conjunto em 1e6 = {lista[-1] / conj[-1]:,.0f}; laço/NumPy = {vezes:.0f}")


blocos = PADRAO.findall(TEX)
assert len(blocos) == 6, f"esperava 6 pares código/saída, achei {len(blocos)}"
for script, colado in blocos:
    longas = [l for l in Path(script).read_text(encoding="utf8").splitlines() if len(l) > 66]
    assert not longas, f"{script}: linha com mais de 66 caracteres: {longas[0]}"
    r = subprocess.run([sys.executable, "-W", "error", script], capture_output=True, text=True)
    assert r.returncode == 0, f"{script} falhou:\n{r.stderr}"
    if script == "l06_custo.py":
        gravada = Path("l06_custo_saida.txt").read_text(encoding="utf8")
        assert sem_espaco_final(colado) == sem_espaco_final(gravada), "l06: colado difere da gravação"
        confere_tempos(gravada, "gravada")
        t = [list(map(float, l.split())) for l in gravada.splitlines()[1:5]]
        assert t[-1][1] / t[-1][2] > 2e5, "texto: 'mais de 200 mil buscas'"
        assert all(0.015 <= x[2] <= 0.025 for x in t), "texto: 'em torno de 0,02'"
        confere_tempos(r.stdout, "nova")
    else:
        assert sem_espaco_final(colado) in sem_espaco_final(r.stdout), f"{script}: saída colada difere da execução"
    print(f"ok  {script}: {len(colado.strip().splitlines())} linhas de saída conferidas")
print("todas as saídas do capítulo conferem")
