"""Testes de sanidade de carteira.py e experimento.py."""
import numpy as np

from carteira import (ACOES, carregar_cdi, cdi_anual, insumos,
                      maximo_sharpe, minima_variancia, otimizar,
                      retornos, vol)
from experimento import JANELA, simular

# 1. dois ativos independentes: pesos proporcionais a 1/var
cov2 = np.diag([0.04, 0.09])
w = minima_variancia(cov2)
assert np.allclose(w, [9 / 13, 4 / 13], atol=1e-6)

# 2. condição de ótimo com pesos não negativos: ativos na
#    carteira têm a mesma covariância com ela; os zerados, maior
r = retornos("2015", "2024")
mu, cov = insumos(r)
w = minima_variancia(cov)
cov_com = cov @ w
dentro = w > 1e-6
assert np.allclose(cov_com[dentro], w @ cov @ w, rtol=1e-4)
assert (cov_com[~dentro] > w @ cov @ w).all()

# 3. máximo Sharpe: nenhum de 100 mil sorteios o supera
rf = cdi_anual("2015", "2024")
w = maximo_sharpe(mu, cov, rf)
W = np.random.default_rng(0).dirichlet(np.ones(len(mu)), 100_000)
sorteios = (W @ mu - rf) / np.sqrt(
    np.einsum("ij,jk,ik->i", W, cov, W))
otimo = (w @ mu - rf) / vol(w, cov)
assert sorteios.max() <= otimo
print(f"Sharpe ótimo {otimo:.4f}; melhor sorteio "
      f"{sorteios.max():.4f}")

# 4. primeiro mês de `simular` para 1/N refeito por laço
r = retornos("2012", "2025")
cdi = carregar_cdi(r.index)
diario, _ = simular(r, cdi)
mes = diario.index[0].to_period("M")
dias = r[r.index.to_period("M") == mes]
valor, serie = np.full(7, 1 / 7), []
for _, linha in dias.iterrows():
    novo = valor * (1 + linha.values)
    serie.append(novo.sum() / valor.sum() - 1)
    valor = novo
assert np.allclose(diario["1/N"].iloc[:len(dias)], serie)
assert r.index.get_loc(diario.index[0]) >= JANELA

# 5. casos degenerados: teto inviável dá erro legível; sem
#    ativo acima do CDI, máximo Sharpe vira mínima variância
try:
    otimizar(lambda w: w @ w, 2, teto=0.4)
    raise AssertionError("deveria ter falhado")
except ValueError as erro:
    print("erro esperado:", erro)
w = maximo_sharpe(np.array([0.01, 0.015]), cov2, 0.05)
assert np.allclose(w, minima_variancia(cov2))

# 6. com teto, um ativo acima do CDI não basta: a melhor
#    carteira viável rende 0,5 x 8% + 0,5 x 1% = 4,5% < 5%
w = maximo_sharpe(np.array([0.08, 0.01, 0.01]),
                  np.diag([0.04, 0.09, 0.16]), 0.05, teto=0.5)
assert np.allclose(w, minima_variancia(
    np.diag([0.04, 0.09, 0.16]), 0.5))
#    e, se a melhor viável supera o CDI, a regra otimiza
w = maximo_sharpe(np.array([0.08, 0.06, 0.01]),
                  np.diag([0.04, 0.09, 0.16]), 0.05, teto=0.5)
assert np.allclose(w, [0.5, 0.5, 0], atol=1e-5)
# 7. prêmio positivo só no último ativo e negativo em 1/N:
#    partindo de pesos iguais o SLSQP parava em ótimo local
mu7 = np.array([0.138, 0.057, 0.003, 0.093, 0.210])
w = maximo_sharpe(mu7, np.diag([0.04, 0.05, 0.03, 0.06, 0.09]),
                  0.203)
assert w @ mu7 > 0.203 and np.allclose(w, [0, 0, 0, 0, 1],
                                       atol=1e-6)
print("7 testes passaram")
