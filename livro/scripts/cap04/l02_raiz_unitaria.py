import warnings
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss
# o kpss avisa quando o p-valor cai fora da faixa tabelada
warnings.simplefilter("ignore", InterpolationWarning)
precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
log_p = np.log(ibov)               # log do índice
r = log_p.diff().dropna()          # log-retorno: 1ª diferença
print("série        teste  estat.  crít. 5%  defas.")
for nome, s in [("log do preço", log_p), ("log-retorno ", r)]:
    est, pv, lags, n, crit, _ = adfuller(s, regression="c",
                                         autolag="AIC")
    c5 = crit["5%"]
    print(f"{nome} ADF   {est:7.2f}  {c5:8.2f}  {lags:6d}")
    est, pv, lags, crit = kpss(s, regression="c", nlags="auto")
    c5 = crit["5%"]
    print(f"{nome} KPSS  {est:7.2f}  {c5:8.2f}  {lags:6d}")
est, pv, _, _, crit, _ = adfuller(log_p, regression="ct",
                                     autolag="AIC")
print(f"log do preço, ADF com tendência: {est:.2f}"
      f"  (crít. 5%: {crit['5%']:.2f})")
