"""WTI, Brent e gás: raiz unitária e cointegração (semanal)."""
import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, coint, kpss
from futuros import ler

warnings.simplefilter("ignore")      # avisos de tabela do KPSS
p = ler(["wti", "brent", "hh"], "2008").dropna()
p = p[(p > 0).all(axis=1)]            # tira 20/04/2020 (WTI < 0)
sem = np.log(p).resample("W-FRI").last().dropna()
sem["dif"] = sem.brent - sem.wti         # ln(Brent/WTI)
for c in sem:
    print(f"{c:5s} ADF p = {adfuller(sem[c])[1]:.3f}"
          f"   KPSS p = {kpss(sem[c], nlags='auto')[1]:.3f}")
for ini, fim in [("2008", "2026"), ("2008", "2016"),
                 ("2017", "2026")]:
    x = sem.loc[ini:fim]
    pv = [coint(x.wti, x[c])[1] for c in ("brent", "hh")]
    print(f"Engle-Granger {ini}-{fim}: WTI x Brent p = "
          f"{pv[0]:.3f}, WTI x gás p = {pv[1]:.3f}")
phi = np.polyfit(sem.dif.shift().dropna(), sem.dif.iloc[1:], 1)[0]
print(f"ln(Brent/WTI): phi = {phi:.3f}, meia-vida "
      f"{np.log(0.5) / np.log(phi):.1f} semanas")
