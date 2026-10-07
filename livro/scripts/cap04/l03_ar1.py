import numpy as np
from statsmodels.tsa.stattools import acf
c, phi, sigma, T = 0.5, 0.8, 1.0, 200_000
rng = np.random.default_rng(7)
eps = rng.normal(0, sigma, T)
y = np.empty(T)
y[0] = c / (1 - phi)               # começa na média teórica
for t in range(1, T):
    y[t] = c + phi * y[t - 1] + eps[t]
print(f"média:     simulada {y.mean():.3f}   "
      f"teórica {c / (1 - phi):.3f}")
print(f"variância: simulada {y.var():.3f}   "
      f"teórica {sigma**2 / (1 - phi**2):.3f}")
print("ACF simulada, k = 1..4:", acf(y, nlags=4)[1:].round(3))
print("phi^k,        k = 1..4:", phi ** np.arange(1, 5))
