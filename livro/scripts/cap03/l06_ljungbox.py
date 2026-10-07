import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()

series = {"retornos": r, "retornos ao quadrado": r ** 2}
for nome, x in series.items():
    rho = acf(x, nlags=3)[1:]
    print(f"{nome}: autocorrelações 1 a 3 =", rho.round(3))
    print(acorr_ljungbox(x, lags=[5, 10, 20]).round(4))

faixa = 1.96 / np.sqrt(len(r))
print(f"faixa de 95% sob independência: ±{faixa:.3f}")
