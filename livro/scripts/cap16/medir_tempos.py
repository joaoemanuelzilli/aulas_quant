"""Mede tempos de execução para o Capítulo 16 e grava dados/c16_tempos.csv.

Cada tempo é o menor de cinco repetições (timeit), por chamada.
O menor tempo é o que mais se aproxima do custo do código em si;
os demais incluem interrupções do sistema. Os números dependem da
máquina: o que se compara são razões e inclinações, não segundos.
Rode com OMP_NUM_THREADS=2. Não regrava o arquivo se ele existe,
para que o texto e a figura usem sempre a mesma medição
(apague-o para medir de novo).
"""
import bisect
import math
import platform
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

from algoritmos import (drawdown_maximo, drawdown_maximo_ingenuo,
                        media_movel, media_movel_ingenua,
                        media_movel_np, melhor_negocio,
                        melhor_negocio_ingenuo)

DESTINO = Path(__file__).resolve().parents[2] / "dados" / "c16_tempos.csv"


def tempo(f, repeticoes=5):
    t = timeit.Timer(f)
    numero, _ = t.autorange()       # chamadas suficientes para 0,2 s
    return min(t.repeat(repeticoes, numero)) / numero


def passeio(n, semente=2026):
    rng = np.random.default_rng(semente)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def log_ret_laco(p):
    r = []
    for i in range(1, len(p)):
        r.append(math.log(p[i] / p[i - 1]))
    return r


def log_ret_compreensao(p):
    return [math.log(p[i] / p[i - 1]) for i in range(1, len(p))]


def log_ret_numpy(p):
    return np.log(p[1:] / p[:-1])


def medir():
    linhas = []

    def anota(exp, metodo, n, seg):
        linhas.append((exp, metodo, n, seg))
        print(f"{exp:12s} {metodo:12s} n={n:>9d}  {seg:.3e} s")

    # 1. melhor compra e venda: O(n^2) contra O(n)
    for n in (125, 250, 500, 1000, 2000, 4000):
        p = list(passeio(n))
        anota("negocio", "ingenuo", n,
              tempo(lambda: melhor_negocio_ingenuo(p)))
    for n in (125, 250, 500, 1000, 2000, 4000, 10**4, 10**5, 10**6):
        p = list(passeio(n))
        anota("negocio", "uma_passada", n,
              tempo(lambda: melhor_negocio(p)))

    # 2. os mesmos algoritmos nos 3.967 pregões de BOVA11
    px = pd.read_csv(DESTINO.parent / "precos_ajustados.csv",
                     index_col=0, parse_dates=True)["BOVA11.SA"]
    v = list(px.dropna())
    n = len(v)
    anota("bova11", "negocio_ing", n,
          tempo(lambda: melhor_negocio_ingenuo(v), 3))
    anota("bova11", "negocio", n, tempo(lambda: melhor_negocio(v)))
    anota("bova11", "dd_ingenuo", n,
          tempo(lambda: drawdown_maximo_ingenuo(v), 3))
    anota("bova11", "dd", n, tempo(lambda: drawdown_maximo(v)))
    anota("bova11", "mm_ingenua", n,
          tempo(lambda: media_movel_ingenua(v, 252)))
    anota("bova11", "mm", n, tempo(lambda: media_movel(v, 252)))
    a = np.array(v)
    anota("bova11", "mm_numpy", n, tempo(lambda: media_movel_np(a, 252)))

    # 3. log-retornos: laço em Python contra NumPy
    for n in (10, 100, 1000, 10**4, 10**5, 10**6, 10**7):
        a = passeio(n)
        lst = a.tolist()
        if n <= 10**6:
            anota("logret", "laco", n, tempo(lambda: log_ret_laco(lst)))
            anota("logret", "compreensao", n,
                  tempo(lambda: log_ret_compreensao(lst)))
        anota("logret", "numpy", n, tempo(lambda: log_ret_numpy(a)))

    # 4. procurar um valor ausente em 10^6 elementos
    n = 10**6
    lst = list(range(0, 2 * n, 2))          # pares, já ordenados
    conj = set(lst)
    alvo = 2 * n + 1                        # não está em nenhum
    anota("busca", "lista_in", n, tempo(lambda: alvo in lst))
    anota("busca", "set_in", n, tempo(lambda: alvo in conj))
    anota("busca", "bisect", n,
          tempo(lambda: bisect.bisect_left(lst, alvo)))
    return pd.DataFrame(linhas,
                        columns=["experimento", "metodo", "n", "segundos"])


if __name__ == "__main__":
    print(platform.processor() or platform.machine(),
          "| Python", sys.version.split()[0], "| NumPy", np.__version__)
    if DESTINO.exists():
        print("já existe:", DESTINO.name, "(apague para medir de novo)")
    else:
        df = medir()
        df.to_csv(DESTINO, index=False)
        print("gravado:", DESTINO)
