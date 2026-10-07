import numpy as np
from carteira import R, w, V

Ra = R["2015":"2024"]
mu, Sigma = Ra.mean().values, Ra.cov().values

rng = np.random.default_rng(42)
N = 200_000
C = np.linalg.cholesky(Sigma)       # Sigma = C C^T
Z = rng.standard_normal((N, 5))     # normais independentes
X = Z @ C.T                         # covariância Sigma

L_normal = -V * ((mu + X) @ w)

# t multivariada com nu graus de liberdade e mesma covariância
nu = 4.48
g = rng.chisquare(nu, N) / nu
X_t = X * np.sqrt((nu - 2) / nu) / np.sqrt(g)[:, None]
L_t = -V * ((mu + X_t) @ w)

for nome, L in (("normal", L_normal), ("t", L_t)):
    print(f"Monte Carlo {nome:6s}  VaR 95%: "
          f"R$ {np.quantile(L, 0.95):,.0f}   VaR 99%: "
          f"R$ {np.quantile(L, 0.99):,.0f}")
