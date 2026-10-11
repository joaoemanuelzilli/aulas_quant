import numpy as np
import pandas as pd
from scipy import stats

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()
n, media, s = len(r), r.mean(), r.std()

ep = s / np.sqrt(n)                    # erro-padrão da média
t = media / ep                         # H0: média igual a zero
p = 2 * stats.t.sf(abs(t), df=n - 1)   # p-valor bilateral
print(f"n = {n} ({n / 252:.1f} anos)   média = {media:.4%} ao dia"
      f" = {252 * media:.2%} ao ano")
print(f"desvio = {s:.4%}   erro-padrão = {ep:.4%}")
vol = s * np.sqrt(252)
print(f"vol. anual = {vol:.1%}   média/vol. anual = "
      f"{252 * media / vol:.3f}")
print(f"t = {t:.3f}   p-valor = {p:.3f}")
res = stats.ttest_1samp(r, 0)
print(f"scipy: t = {res.statistic:.3f}"
      f"   p-valor = {res.pvalue:.3f}")

ic = media + np.array([-1, 1]) * stats.t.ppf(0.975, n - 1) * ep
print(f"IC de 95% para a média anual: {252 * ic[0]:.1%}"
      f" a {252 * ic[1]:.1%}")

# dias necessários se a média verdadeira fosse a amostral
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
for nome, z in [("t = 1,96", z_a), ("poder de 80%", z_a + z_b)]:
    dias = (z * s / media) ** 2
    anos = dias / 252
    print(f"{nome:13s}: {dias:6.0f} dias = {anos:4.0f} anos")

# a pergunta do investidor: o índice rende mais do que o CDI?
bcb = pd.read_csv("../../dados/c02_bcb.csv",
                  index_col=0, parse_dates=True)
cdi_ac = np.log1p(bcb["cdi"] / 100).cumsum()  # CDI em % ao dia
cdi = cdi_ac.reindex(ibov.index, method="ffill").diff()
exc = (r - cdi).dropna()               # log-retorno em excesso
t_exc = exc.mean() / (exc.std() / np.sqrt(len(exc)))
print(f"CDI = {252 * cdi.mean():.2%} ao ano   excesso = "
      f"{252 * exc.mean():.2%} ao ano   t = {t_exc:.3f}")
