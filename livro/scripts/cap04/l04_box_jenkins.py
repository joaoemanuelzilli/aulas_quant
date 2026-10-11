import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = 100 * np.log(ibov).diff().dropna()      # log-retornos em %
# sem as datas: pregões não têm frequência fixa e o ARIMA avisa
treino = r[:"2021"].to_numpy()              # 2010 a 2021
print("ordem          AIC      BIC   p (Ljung-Box)")
ordens = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 2)]
for ordem in ordens:
    m = ARIMA(treino, order=ordem).fit()
    # diagnóstico: resta autocorrelação nos resíduos?
    lb = acorr_ljungbox(m.resid, lags=[10],
                        model_df=ordem[0] + ordem[2])
    print(f"{ordem}  {m.aic:.1f}  {m.bic:.1f}"
          f"   {lb.lb_pvalue[10]:.4f}")
ar1 = ARIMA(treino, order=(1, 0, 0)).fit(cov_type="robust")
print(f"AR(1): phi = {ar1.params[1]:.4f}"
      f"   erro-padrão robusto = {ar1.bse[1]:.4f}")
lb2 = acorr_ljungbox(ar1.resid**2, lags=[10])
print(f"Ljung-Box dos resíduos ao quadrado: "
      f"Q = {lb2.lb_stat[10]:.0f}")
