import numpy as np

from carteira import insumos, retornos

r = retornos("2015", "2024")
_, cov = insumos(r)
n = len(cov)
nomes = [c[:5] for c in r.columns]

# 1. equilíbrio: retornos em excesso que tornam ótima a
#    carteira de referência (aqui 1/N, por falta de valores
#    de mercado no conjunto de dados; é uma hipótese)
w_ref = np.full(n, 1 / n)
delta, tau = 2.5, 0.05
pi = delta * cov @ w_ref

# 2. uma visão relativa: WEGE3 rende 5 p.p. a mais que ITUB4
P = np.zeros((1, n))
P[0, nomes.index("WEGE3")], P[0, nomes.index("ITUB4")] = 1, -1
q = np.array([0.05])
omega = P @ (tau * cov) @ P.T     # incerteza da visão

# 3. média a posteriori
A = np.linalg.inv(tau * cov) + P.T @ np.linalg.inv(omega) @ P
b = (np.linalg.inv(tau * cov) @ pi
     + P.T @ np.linalg.inv(omega) @ q)
mu_bl = np.linalg.solve(A, b)
w_bl = np.linalg.solve(delta * cov, mu_bl)

print("        ", "  ".join(nomes))
for rot, v in [("pi (%)   ", pi), ("mu_BL (%)", mu_bl),
               ("w_ref (%)", w_ref), ("w_BL (%) ", w_bl)]:
    print(rot, "  ".join(f"{x * 100:5.1f}" for x in v))
print(f"visão embutida no equilíbrio: {(P @ pi)[0]:.2%}; "
      f"a posteriori: {(P @ mu_bl)[0]:.2%}")
