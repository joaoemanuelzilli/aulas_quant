"""Testes de sanidade das funções do capítulo. Rode depois de
qualquer alteração:  python teste_risco.py"""
import numpy as np
from scipy import stats
from carteira import R, Rp, w, perda, cota, var_historico
from l05_kupiec import kupiec, christoffersen

# Kupiec contra a conta por extenso e nos casos-limite
x, T, p = 7, 250, 0.01
f = x / T
lr = -2 * (x * np.log(p / f) + (T - x) * np.log((1 - p) / (1 - f)))
assert np.isclose(kupiec(x, T, p)[0], lr)
assert np.isclose(kupiec(0, 250, 0.01)[0], -2 * 250 * np.log(0.99))
assert np.isclose(kupiec(5, 500, 0.01), (0, 1)).all()

# Christoffersen: violações alternadas são dependentes; ao acaso, não
import pandas as pd
rng = np.random.default_rng(0)
assert christoffersen(pd.Series([0, 1] * 500))[1] < 1e-6
assert christoffersen(pd.Series(rng.random(20000) < 0.05))[1] > 0.01

# w'Σw é a variância da série da carteira
assert np.isclose(w @ R.cov().values @ w, Rp.var())

# sem look-ahead: mudar a perda de um dia não muda o VaR desse dia
p2 = perda.copy(); d = p2.index[1000]; p2[d] = 1e9
assert var_historico(p2)[d] == var_historico(perda)[d]
assert var_historico(p2).iloc[1001] != var_historico(perda).iloc[1001]

# drawdown por laço
pico, pior = -np.inf, 0.0
for v in cota:
    pico = max(pico, v)
    pior = min(pior, v / pico - 1)
assert np.isclose(pior, (cota / cota.cummax() - 1).min())

# VaR e ES da normal contra sorteios
z = rng.standard_normal(2_000_000)
q = np.quantile(z, 0.99)
assert abs(q - stats.norm.ppf(0.99)) < 0.01
za = stats.norm.ppf(0.99)
assert abs(z[z >= q].mean() - stats.norm.pdf(za) / 0.01) < 0.02
print("todos os testes passaram")
