import numpy as np
import pandas as pd
from scipy import stats

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()
n, mu, sigma = len(r), r.mean(), r.std()

nu, loc, escala = stats.t.fit(r)       # máxima verossimilhança
t_aj = stats.t(nu, loc, escala)
print(f"amostra:    n = {n}  média = {mu:.5f}"
      f"  desvio = {sigma:.5f}")
print(f"t ajustada: nu = {nu:.2f}  loc = {loc:.5f}"
      f"  escala = {escala:.5f}  desvio = {t_aj.std():.5f}")

print(" k   observado   normal   t ajustada")
for k in [2, 3, 4, 5]:
    obs = (np.abs(r - mu) > k * sigma).sum()
    esp_n = n * 2 * stats.norm.sf(k)
    esp_t = n * (t_aj.cdf(mu - k * sigma)
                 + t_aj.sf(mu + k * sigma))
    print(f"{k:2d}   {obs:9d}   {esp_n:6.2f}   {esp_t:10.2f}")
