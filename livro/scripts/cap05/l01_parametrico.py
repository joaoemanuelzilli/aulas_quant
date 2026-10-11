import numpy as np
from scipy import stats
from carteira import R, w, V, perda

Ra = R["2015":"2024"]
mu = Ra.mean().values          # médias diárias dos 5 ativos
Sigma = Ra.cov().values        # matriz de covariância 5 x 5

mu_L = -V * (w @ mu)                    # média da perda
sig_L = V * np.sqrt(w @ Sigma @ w)      # desvio-padrão da perda
assert np.isclose(sig_L, perda["2015":"2024"].std())
print(f"mu_L = {mu_L:,.0f}   sigma_L = {sig_L:,.0f}")

for a in (0.95, 0.99):
    z = stats.norm.ppf(a)
    print(f"VaR normal {a:.0%}: z = {z:.3f}  "
          f"R$ {mu_L + z * sig_L:,.0f}")

# VaR de cada posição isolada e ganho de diversificação
sig_i = V * w * np.sqrt(np.diag(Sigma))
z = stats.norm.ppf(0.99)
print("soma dos VaR 99% isolados (média zero): "
      f"R$ {z * sig_i.sum():,.0f}")
print(f"VaR 99% da carteira (média zero):       "
      f"R$ {z * sig_L:,.0f}")
