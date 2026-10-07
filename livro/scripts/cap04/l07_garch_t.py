import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = 100 * np.log(ibov).diff().dropna()
especs = {
    "GARCH normal": dict(p=1, q=1, dist="normal"),
    "GARCH t     ": dict(p=1, q=1, dist="t"),
    "GJR normal  ": dict(p=1, o=1, q=1, dist="normal"),
    "GJR t       ": dict(p=1, o=1, q=1, dist="t"),
}
print("modelo        log-veros.      AIC      BIC      mu")
ajustes = {}
for nome, kw in especs.items():
    m = arch_model(r, vol="GARCH", **kw).fit(disp="off")
    ajustes[nome] = m
    print(f"{nome}  {m.loglikelihood:10.1f}"
          f"  {m.aic:.1f}  {m.bic:.1f}  {m.params['mu']:.4f}")
z = ajustes["GARCH normal"].std_resid      # resíduos padronizados
k_r = stats.kurtosis(r, fisher=False)
k_z = stats.kurtosis(z, fisher=False)
print(f"curtose: retornos {k_r:.2f}"
      f"   resíduos padronizados {k_z:.2f}")
print(f"|z| > 3: {np.mean(abs(z) > 3):.2%} dos dias"
      f"   (normal: {2 * stats.norm.sf(3):.2%})")
print(f"graus de liberdade da t: "
      f"{ajustes['GARCH t     '].params['nu']:.1f}")
gjr = ajustes["GJR t       "].params
print(f"GJR t: alfa {gjr['alpha[1]']:.3f}"
      f"   gama {gjr['gamma[1]']:.3f}"
      f"   beta {gjr['beta[1]']:.3f}")
