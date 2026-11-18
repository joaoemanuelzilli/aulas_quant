import numpy as np

from carteira import NOMES, insumos, retornos

r = retornos("2015", "2024")
_, cov = insumos(r)
n = len(cov)

# 1. equilíbrio: retornos em excesso que tornam ótima a
#    carteira de referência (aqui 1/N, por falta de valores
#    de mercado no conjunto de dados; é uma hipótese)
w_ref = np.full(n, 1 / n)
premio, tau = 0.06, 0.05           # prêmio suposto para w_ref
delta = premio / (w_ref @ cov @ w_ref)
pi = delta * cov @ w_ref

# 2. uma visão relativa: WEGE3 rende 5 p.p. a mais que ITUB4
P = np.zeros((1, n))
P[0, NOMES.index("WEGE3")], P[0, NOMES.index("ITUB4")] = 1, -1
q = np.array([0.05])
omega = P @ (tau * cov) @ P.T     # incerteza da visão

# 3. média a posteriori (Equação 6.11) e pesos ótimos
inv_tc = np.linalg.inv(tau * cov)
inv_om = np.linalg.inv(omega)
mu_bl = np.linalg.solve(inv_tc + P.T @ inv_om @ P,
                        inv_tc @ pi + P.T @ inv_om @ q)
w_bl = np.linalg.solve(delta * cov, mu_bl)

print(f"delta = {delta:.2f}")
print("        ", "  ".join(NOMES))
for rot, v in [("pi (%)   ", pi), ("mu_BL (%)", mu_bl),
               ("w_ref (%)", w_ref), ("w_BL (%) ", w_bl)]:
    print(rot, "  ".join(f"{x * 100:5.1f}" for x in v))
print(f"visão embutida no equilíbrio: {(P @ pi)[0]:.2%}; "
      f"a posteriori: {(P @ mu_bl)[0]:.2%}")

# conferência pela forma fechada de uma única visão
psp = (P @ cov @ P.T)[0, 0]
ajuste = (q[0] - (P @ pi)[0]) / (2 * delta * psp)
print(f"P Sigma P' = {psp:.4f}; ajuste dos pesos: {ajuste:.1%}")
assert np.allclose(w_bl, w_ref + ajuste * P[0])
