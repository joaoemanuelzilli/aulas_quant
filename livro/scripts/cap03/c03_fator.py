"""Conferência das contas feitas à mão na seção de correlação."""
import numpy as np

rng = np.random.default_rng(3)
n = 1_000_000
for sm in [0.01, 0.04]:                 # volatilidade do fator comum
    m = rng.normal(0, sm, n)
    a = m + rng.normal(0, 0.015, n)     # dois ativos: fator + ruído
    b = m + rng.normal(0, 0.015, n)
    teorico = sm ** 2 / (sm ** 2 + 0.015 ** 2)
    print(f"vol. do fator {sm:.0%}: correlação teórica {teorico:.4f}"
          f"  simulada {np.corrcoef(a, b)[0, 1]:.4f}")

# erro-padrão aproximado da correlação amostral: (1 - rho^2)/raiz(n)
for rho, k in [(0.3, 63), (0.3, 252), (0.3, 2462)]:
    ep = (1 - rho ** 2) / np.sqrt(k)
    sim = [np.corrcoef(rng.multivariate_normal(
        [0, 0], [[1, rho], [rho, 1]], k).T)[0, 1] for _ in range(4000)]
    print(f"rho={rho} n={k}: erro-padrão {ep:.4f}"
          f"  simulado {np.std(sim):.4f}  ±1,96 ep = {1.96 * ep:.3f}")

# relações válidas para a normal bivariada
rho = 0.443
print("Kendall implícito:", 2 / np.pi * np.arcsin(rho))
print("Spearman implícito:", 6 / np.pi * np.arcsin(rho / 2))
print("pares distintos: N=7 ->", 7 * 6 // 2, " N=100 ->", 100 * 99 // 2,
      " parâmetros de Sigma com N=100:", 100 * 101 // 2)
