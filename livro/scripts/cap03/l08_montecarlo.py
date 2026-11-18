import numpy as np
from scipy.stats import norm

mu, sigma, P0 = 0.10, 0.48, 30.0       # o modelo lognormal
exato = norm.cdf((np.log(0.70) - mu) / sigma)

rng = np.random.default_rng(seed=2026)  # semente fixa
for N in (100, 10_000, 1_000_000):
    r = rng.normal(mu, sigma, size=N)   # N log-retornos anuais
    PT = P0 * np.exp(r)                 # N preços daqui a um ano
    p = np.mean(PT < 0.70 * P0)         # fração com queda > 30%
    ep = np.sqrt(p * (1 - p) / N)       # erro-padrão
    print(f"N = {N:>9,}  P ~ {p:.4f} ± {ep:.4f}"
          f"   E[P_T] ~ {PT.mean():.2f}")
media = P0 * np.exp(mu + sigma**2 / 2)
print(f"exato          P = {exato:.4f}"
      f"            E[P_T] = {media:.2f}")
