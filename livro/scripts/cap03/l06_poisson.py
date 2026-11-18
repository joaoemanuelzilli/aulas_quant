import numpy as np
import pandas as pd
from scipy.stats import poisson

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()
queda = r < -0.03                       # dia de queda extrema
N = queda.groupby(r.index.to_period("M")).sum()  # por mês

lam = N.mean()
print(f"{queda.sum()} dias em {len(N)} meses: média {lam:.3f}"
      f"   variância {N.var():.3f}")
print(" k   meses   Poisson")
for k in range(4):
    esperado = len(N) * poisson.pmf(k, lam)
    print(f"{k:2d}  {np.sum(N == k):6d}  {esperado:8.1f}")
esperado = len(N) * poisson.sf(3, lam)  # P(N >= 4)
print(f">=4 {np.sum(N >= 4):6d}  {esperado:8.1f}")
print("3 ou mais:", " ".join(str(m) for m in N[N >= 3].index))
