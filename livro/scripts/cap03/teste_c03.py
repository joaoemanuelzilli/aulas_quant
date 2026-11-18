"""Testes das contas centrais do Capítulo 3.

Rodar de dentro de scripts/cap03:  python teste_c03.py
Cada teste confere uma fórmula do texto por um caminho independente
(enumeração, simulação com semente fixa ou a implementação do SciPy).
"""
from itertools import product
from math import comb, exp, factorial

import numpy as np
from scipy.stats import binom, lognorm, norm, poisson

from l02_bayes import posterior

rng = np.random.default_rng(seed=123)

# 1. regra de Bayes = contagem direta em uma população simulada
n = 2_000_000
H = rng.random(n) < 0.0192
S = np.where(H, rng.random(n) < 0.80, rng.random(n) < 0.10)
assert abs(H[S].mean() - posterior(0.0192, 0.80, 0.10)) < 0.003
assert np.isclose(posterior(0.5, 0.3, 0.3), 0.5)        # sinal inútil
assert np.isclose(posterior(0.2, 1.0, 0.0), 1.0)        # sinal perfeito

# 2. lei da probabilidade total e Bayes por enumeração: dois dados
pares = list(product(range(1, 7), repeat=2))
seis = [p for p in pares if 6 in p]
assert len(seis) == 11
assert sum(a + b == 7 for a, b in seis) == 2           # P(7 | um 6) = 2/11

# 3. binomial: fórmula = SciPy; média np e variância np(1-p)
for nn, p in ((5, 0.5), (21, 0.5134), (250, 0.01)):
    k = np.arange(nn + 1)
    f = np.array([comb(nn, i) * p**i * (1 - p)**(nn - i) for i in k])
    assert np.allclose(f, binom.pmf(k, nn, p))
    assert np.isclose((k * f).sum(), nn * p)
    assert np.isclose(((k - nn * p)**2 * f).sum(), nn * p * (1 - p))

# 4. Poisson: fórmula = SciPy, média = variância, limite da binomial
lam = 0.396
k = np.arange(60)
f = np.array([exp(-lam) * lam**i / factorial(i) for i in k])
assert np.allclose(f, poisson.pmf(k, lam))
assert np.isclose((k * f).sum(), lam)
assert np.isclose(((k - lam)**2 * f).sum(), lam)
assert abs(binom.pmf(3, 100_000, 2.5 / 100_000) - poisson.pmf(3, 2.5)) < 1e-5

# 5. variância de uma soma: fórmula = simulação de normais correlacionadas
sx, sy, rho, w = 0.03, 0.025, 0.44, 0.5
C = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
Z = rng.multivariate_normal([0, 0], C, size=1_000_000)
vp = w**2 * sx**2 + (1 - w)**2 * sy**2 + 2 * w * (1 - w) * rho * sx * sy
assert abs((Z @ [w, 1 - w]).var() / vp - 1) < 0.01

# 6. normal: padronizar não muda a probabilidade; regra 68-95-99,7
assert np.isclose(norm.cdf(-0.2, 0.1, 0.48), norm.cdf((-0.2 - 0.1) / 0.48))
assert np.allclose([2 * norm.cdf(z) - 1 for z in (1, 2, 3)],
                   [0.6827, 0.9545, 0.9973], atol=5e-5)

# 7. lognormal: mediana e média da Equação (3.13) = SciPy = simulação
mu, s, P0, T = 0.10, 0.48, 30.0, 5
d = lognorm(s=s * np.sqrt(T), scale=P0 * np.exp(mu * T))
assert np.isclose(d.median(), P0 * np.exp(mu * T))
assert np.isclose(d.mean(), P0 * np.exp(mu * T + s**2 * T / 2))
PT = P0 * np.exp(rng.normal(mu, s, 2_000_000))
assert abs(PT.mean() / (P0 * np.exp(mu + s**2 / 2)) - 1) < 0.002

# 8. Monte Carlo: o erro-padrão sqrt(p(1-p)/N) descreve a dispersão real
p = norm.cdf((np.log(0.7) - mu) / s)
N = 2_000
est = np.array([np.mean(rng.normal(mu, s, N) < np.log(0.7))
                for _ in range(2_000)])
assert abs(est.std() / np.sqrt(p * (1 - p) / N) - 1) < 0.05
assert abs(np.mean(np.abs(est - p) < 1.96 * est.std()) - 0.95) < 0.02

# 9. aposta: E[W_n] = 1 e P(W_10 > 1) = P(K >= 7)
k = np.arange(11)
W = 1.5**k * 0.5**(10 - k)
assert np.isclose((W * binom.pmf(k, 10, 0.5)).sum(), 1.0)
assert np.isclose(binom.pmf(k[W > 1], 10, 0.5).sum(), 176 / 1024)

print("todos os testes do Capítulo 3 passaram")
