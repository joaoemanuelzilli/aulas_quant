"""Problemas de combinatória do Capítulo 16: fórmula e força bruta.

Cada resposta fechada é conferida contando os casos um a um
(ou, quando são muitos, por simulação com semente fixa).
"""
import itertools
import math
from collections import Counter

import numpy as np


# 1. caminhos em uma árvore binomial (ou em uma grade)
def caminhos(n, k):
    return math.comb(n, k)


def caminhos_forca_bruta(n, k):
    return sum(1 for s in itertools.product((0, 1), repeat=n)
               if sum(s) == k)


def caminhos_grade(m, n):
    """Programação dinâmica: c[i][j] = c[i-1][j] + c[i][j-1]."""
    c = [[1] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c[i][j] = c[i - 1][j] + c[i][j - 1]
    return c[m][n]


# 2. estrelas e barras: lotes inteiros entre ativos
def estrelas_barras(n, k, minimo=0):
    n = n - k * minimo
    return math.comb(n + k - 1, k - 1)


def estrelas_barras_forca_bruta(n, k, minimo=0):
    faixa = range(minimo, n + 1)
    return sum(1 for x in itertools.product(faixa, repeat=k)
               if sum(x) == n)


# 3. pôquer: mãos de cinco cartas com exatamente um par
def um_par():
    return 13 * math.comb(4, 2) * math.comb(12, 3) * 4 ** 3


def um_par_forca_bruta():
    """Percorre as 2.598.960 mãos (leva alguns segundos)."""
    valores = [c // 4 for c in range(52)]
    total = 0
    for mao in itertools.combinations(range(52), 5):
        contagem = sorted(Counter(valores[c] for c in mao).values())
        total += contagem == [1, 1, 1, 2]
    return total


# 4. permutações sem ponto fixo (o problema dos chapéus)
def desarranjos(n):
    """Inclusão-exclusão: D_n = n! soma (-1)^k / k!."""
    return sum((-1) ** k * math.factorial(n) // math.factorial(k)
               for k in range(n + 1))


def desarranjos_forca_bruta(n):
    return sum(1 for p in itertools.permutations(range(n))
               if all(p[i] != i for i in range(n)))


def desarranjo_simulado(n, sims=200_000, semente=2026):
    rng = np.random.default_rng(semente)
    p = rng.permuted(np.tile(np.arange(n), (sims, 1)), axis=1)
    return np.mean((p != np.arange(n)).all(axis=1))


if __name__ == "__main__":
    print("caminhos 10 passos, 6 altas:", caminhos(10, 6),
          caminhos_forca_bruta(10, 6), "de", 2 ** 10)
    print("grade 5x5:", caminhos(10, 5), caminhos_grade(5, 5))
    print("10 lotes, 4 ativos:", estrelas_barras(10, 4),
          estrelas_barras_forca_bruta(10, 4))
    print("  com ao menos 1 em cada:", estrelas_barras(10, 4, 1),
          estrelas_barras_forca_bruta(10, 4, 1))
    tot = math.comb(52, 5)
    print("um par:", um_par(), "de", tot, f"= {um_par() / tot:.4f}")
    print("D_8:", desarranjos(8), desarranjos_forca_bruta(8))
    d10 = desarranjos(10)
    print("D_10:", d10, f"p = {d10 / math.factorial(10):.6f}",
          f"1/e = {1 / math.e:.6f}",
          f"simulado = {desarranjo_simulado(10):.4f}")
