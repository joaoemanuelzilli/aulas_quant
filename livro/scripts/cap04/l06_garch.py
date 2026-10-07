import numpy as np
import pandas as pd
from arch import arch_model
precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = 100 * np.log(ibov).diff().dropna()     # retornos em %
modelo = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="normal")
res = modelo.fit(disp="off")
print(f"mu estimado: {res.params['mu']:.4f}%"
      f"   média amostral: {r.mean():.4f}%")
print(res.summary().tables[2])             # equação da variância
omega, alfa, beta = res.params[1:]         # depois de mu
persistencia = alfa + beta
var_inc = omega / (1 - persistencia)       # de longo prazo
meia_vida = np.log(0.5) / np.log(persistencia)
print(f"persistência alfa + beta: {persistencia:.4f}")
print(f"vol. incondicional: {np.sqrt(var_inc):.3f}% ao dia, "
      f"{np.sqrt(252 * var_inc):.1f}% ao ano")
print(f"desvio-padrão amostral: {r.std():.3f}% ao dia")
print(f"meia-vida da variância: {meia_vida:.1f} dias úteis")
vol = res.conditional_volatility * np.sqrt(252)
print(f"vol. condicional anualizada: mínima {vol.min():.1f}%, "
      f"máxima {vol.max():.1f}% em {vol.idxmax():%d/%m/%Y}")
