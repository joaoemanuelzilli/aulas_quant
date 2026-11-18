import numpy as np
import pandas as pd
from scipy.stats import binom

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
petr = precos["PETR4.SA"].dropna()
alta = (petr.pct_change().dropna() > 0).to_numpy()

n = 21                                 # pregões por bloco
blocos = len(alta) // n
K = alta[:blocos * n].reshape(blocos, n).sum(axis=1)
p = alta.mean()
print(f"{len(alta)} pregões, {blocos} blocos de {n},"
      f" p = {p:.4f}")
print(f"dados:    média {K.mean():.2f}   variância "
      f"{K.var():.2f}")
print(f"binomial: média {n * p:.2f}   variância "
      f"{n * p * (1 - p):.2f}")
for k in (7, 14):
    print(f"P(K <= {k:2d}): dados {np.mean(K <= k):.3f}"
          f"   binomial {binom.cdf(k, n, p):.3f}")
