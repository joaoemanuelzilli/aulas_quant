"""Erro-padrão do Sharpe quando os retornos não são iid.

Quatro mil históricos de dez anos, Sharpe verdadeiro de 0,5 ao
ano, em quatro processos: iid, volatilidade agrupada (GARCH) e
autocorrelação de primeira ordem positiva e negativa (AR).
"""
import numpy as np

DIAS, ANOS, N = 252, 10, 4000
T = DIAS * ANOS
MU, SIGMA = 0.5 / DIAS, 1 / np.sqrt(DIAS)   # Sharpe anual de 0,5
rng = np.random.default_rng(42)


def garch(z, alfa=0.10, beta=0.88):
    """GARCH(1,1) com variância incondicional SIGMA ** 2."""
    omega = SIGMA ** 2 * (1 - alfa - beta)
    var = np.full(z.shape[0], SIGMA ** 2)
    e = np.empty_like(z)
    for t in range(z.shape[1]):
        e[:, t] = np.sqrt(var) * z[:, t]
        var = omega + alfa * e[:, t] ** 2 + beta * var
    return MU + e


def ar1(z, rho):
    """AR(1) com média MU e variância incondicional SIGMA ** 2."""
    e = np.empty_like(z)
    e[:, 0] = SIGMA * z[:, 0]
    escala = SIGMA * np.sqrt(1 - rho ** 2)
    for t in range(1, z.shape[1]):
        e[:, t] = rho * e[:, t - 1] + escala * z[:, t]
    return MU + e


casos = {"iid": lambda z: MU + SIGMA * z,
         "GARCH(1,1)": garch,
         "AR(1), rho = +0.1": lambda z: ar1(z, 0.1),
         "AR(1), rho = -0.1": lambda z: ar1(z, -0.1)}
for nome, gera in casos.items():
    r = gera(rng.standard_normal((N, T)))
    sr = r.mean(axis=1) / r.std(axis=1, ddof=1) * np.sqrt(DIAS)
    print(f"{nome:18s} média {sr.mean():.2f}   "
          f"desvio-padrão {sr.std():.3f}")
print(f"fórmula iid: {np.sqrt((1 + 0.5 ** 2 / 504) / ANOS):.3f}")
