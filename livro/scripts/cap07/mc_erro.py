"""Erro de simulação do VaR de 99% por Monte Carlo normal."""
import numpy as np
from scipy import stats
from carteira import R, w, V

Ra = R["2015":"2024"]
mu, Sigma = Ra.mean().values, Ra.cov().values
C = np.linalg.cholesky(Sigma)
exato = (-V * (w @ mu)
         + stats.norm.ppf(0.99) * V * np.sqrt(w @ Sigma @ w))
rng = np.random.default_rng(7)
print(f"valor exato: {exato:,.0f}")
for N in (1_000, 10_000, 100_000):
    est = np.array([
        np.quantile(
            -V * ((mu + rng.standard_normal((N, 5)) @ C.T) @ w),
            0.99)
        for _ in range(200)])
    print(f"{N:7d} cenários: mín {est.min():,.0f}  "
          f"máx {est.max():,.0f}  desvio-padrão "
          f"{est.std(ddof=1):,.0f} "
          f"({est.std(ddof=1) / exato:.1%} do valor exato)")
