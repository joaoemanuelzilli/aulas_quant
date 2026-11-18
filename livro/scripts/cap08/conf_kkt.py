"""Confere as condições de KKT na carteira de mínima variância sem
venda a descoberto (Seção de otimização numérica do Capítulo 8) e a
leitura do multiplicador de Lagrange da carteira de mínima variância
global. Termina com erro se alguma conta do texto deixar de valer.

Rodar de dentro de scripts/cap08:  python conf_kkt.py
"""
import numpy as np

from carteira import NOMES, insumos, minima_variancia, retornos

r = retornos("2015", "2024")
mu, cov = insumos(r)

# 1. mínima variância sem venda a descoberto: KKT
w = minima_variancia(cov)
var_p = w @ cov @ w
cov_com_carteira = cov @ w            # Cov(R_i, R_p) de cada ação
print(f"variância da carteira: {var_p:.4f} (vol {np.sqrt(var_p):.1%})")
for nome, wi, ci in zip(NOMES, w, cov_com_carteira):
    print(f"  {nome}: peso {wi:6.1%}  Cov(R_i, R_p) = {ci:.4f}")
dentro = w > 1e-6
assert dentro.sum() == 4
# ativos com peso positivo: covariância com a carteira = variância dela
assert np.allclose(cov_com_carteira[dentro], var_p, rtol=1e-4)
# ativos zerados: covariância maior (aumentá-los aumentaria a variância)
assert np.all(cov_com_carteira[~dentro] > var_p)
fora = cov_com_carteira[~dentro]
print(f"zerados: Cov de {fora.min():.4f} a {fora.max():.4f}")

# 2. mínima variância global: o multiplicador é a variância e mede
#    quanto a (meia) variância ótima cresce se a soma dos pesos for c
um = np.ones(len(mu))
inv = np.linalg.inv(cov)
lam = 1 / (um @ inv @ um)
w_mv = lam * inv @ um
assert np.isclose(w_mv @ cov @ w_mv, lam)


def otimo(c):
    """Mínimo de 1/2 w'Σw com soma dos pesos = c, resolvendo o
    sistema de Lagrange [Σ 1; 1' 0] [w; -λ] = [0; c]."""
    n = len(mu)
    A = np.block([[cov, um[:, None]], [um[None, :], np.zeros((1, 1))]])
    x = np.linalg.solve(A, np.r_[np.zeros(n), c])
    w_c = x[:n]
    return 0.5 * w_c @ cov @ w_c, -x[n]


h = 1e-4
deriv = (otimo(1 + h)[0] - otimo(1 - h)[0]) / (2 * h)
assert np.isclose(otimo(1)[1], lam)          # mesmo multiplicador
assert np.isclose(deriv, lam, rtol=1e-6)     # e é a derivada do ótimo
print(f"mínima variância global: lambda = {lam:.4f} = variância "
      f"(vol {np.sqrt(lam):.1%}); d(ótimo)/dc = {deriv:.4f}")
print("ok")
