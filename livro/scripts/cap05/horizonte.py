"""Raiz do tempo contra o quantil empírico de perdas em h dias."""
import numpy as np
from carteira import Rp, V

r = Rp["2015":"2024"]
L1 = -V * r
print("h   raiz(h) x VaR 1d   quantil de h dias   razão")
for a in (0.95, 0.99):
    v1 = np.quantile(L1, a)
    for h in (5, 10, 21):
        acum = (1 + r).rolling(h).apply(np.prod, raw=True) - 1
        Lh = (-V * acum).dropna()
        vh = np.quantile(Lh, a)
        topo = Lh[Lh > vh]
        mar20 = topo["2020-03":"2020-04"]
        print(f"{a:.0%} {h:3d} {np.sqrt(h) * v1:12,.0f} {vh:12,.0f}"
              f"   {vh / (np.sqrt(h) * v1):.2f}   janelas acima: "
              f"{len(topo)}, em mar-abr/2020: {len(mar20)}")
print("autocorrelação de ordem 1 dos retornos:", r.autocorr(1))
print("autocorrelação de ordem 1 dos quadrados:", (r ** 2).autocorr(1))
print("média diária x 10:", r.mean() * 10 * V, " raiz(10)*sigma:", np.sqrt(10) * L1.std())
