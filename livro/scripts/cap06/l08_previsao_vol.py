import numpy as np
import pandas as pd
from arch import arch_model
precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = 100 * np.log(ibov).diff().dropna()
res = arch_model(r, vol="GARCH", p=1, q=1).fit(disp="off")
H = 252
prev = res.forecast(horizon=H).variance.iloc[-1].to_numpy()
# a mesma previsão pela fórmula fechada
par = res.params                           # Series com nomes
omega, alfa, beta = par[["omega", "alpha[1]", "beta[1]"]]
assert 0 < alfa + beta < 1, "sem variância de longo prazo"
var_inc = omega / (1 - alfa - beta)
h = np.arange(1, H + 1)
desvio = prev[0] - var_inc                 # no primeiro passo
formula = var_inc + (alfa + beta) ** (h - 1) * desvio
print("maior diferença biblioteca x fórmula:",
      f"{abs(prev - formula).max():.1e}")
print(f"origem: {r.index[-1]:%d/%m/%Y}   "
      f"vol. de longo prazo: {np.sqrt(var_inc):.3f}%")
print(" h   vol. do dia h   vol. acumulada   raiz do tempo")
for k in (1, 5, 21, 63, 252):
    acumulada = np.sqrt(prev[:k].sum())    # retorno de k dias
    ingenua = np.sqrt(k * prev[0])         # amanhã x raiz(k)
    print(f"{k:3d}   {np.sqrt(prev[k - 1]):10.3f}%"
          f"   {acumulada:11.2f}%   {ingenua:10.2f}%")
