"""Números da tabela de caudas: normal contra t padronizada."""
import numpy as np
from scipy import stats

NUS = [3, 5, 10]
print("P(X < -k), X com média 0 e variância 1")
cab = " ".join(f"{'t(%d)' % v:>11s}" for v in NUS)
print(" k      normal " + cab)
for k in [1, 2, 3, 4, 5, 6]:
    linha = [stats.norm.cdf(-k)]
    for nu in NUS:
        c = np.sqrt(nu / (nu - 2))     # T = c * X
        linha.append(stats.t.cdf(-k * c, nu))
    print(f"{k:2d}  " + "  ".join(f"{v:10.3e}" for v in linha))
print("uma queda além de k desvios a cada ... pregões (1/p)")
for k in [2, 3, 4, 5, 6]:
    linha = [1 / stats.norm.cdf(-k)]
    for nu in NUS:
        c = np.sqrt(nu / (nu - 2))
        linha.append(1 / stats.t.cdf(-k * c, nu))
    print(f"{k:2d}  " + "  ".join(f"{v:15,.0f}" for v in linha))
print("frequência de uma queda além de k desvios")
for k in [3, 4, 5, 6]:
    pn = stats.norm.cdf(-k)
    pt = stats.t.cdf(-k * np.sqrt(5 / 3), 5)
    print(f"k={k}: normal 1 em {1/pn:,.0f} dias ="
          f" {1/pn/252:,.1f} anos; t(5) 1 em {1/pt:,.0f} dias ="
          f" {1/pt/252:,.2f} anos; razão {pt/pn:,.1f}")
for nu in [3, 4, 5, 6, 10, 30]:
    v = nu / (nu - 2)
    ek = 6 / (nu - 4) if nu > 4 else float("inf")
    print(f"nu={nu}: Var {v:.4f}  excesso de curtose {ek:.3f}"
          f"  scipy: {stats.t.stats(nu, moments='vk')}")
# faixas centrais da normal e quantis usados no texto
for k in [1, 2, 3]:
    print(f"P(|Z|<{k}) = {1 - 2 * stats.norm.sf(k):.4%}")
print("IQR da normal padrão:", 2 * stats.norm.ppf(0.75))
print("z 97,5%:", stats.norm.ppf(0.975))
print("z 80%:", stats.norm.ppf(0.8))
