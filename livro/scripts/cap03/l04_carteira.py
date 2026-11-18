import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
p = precos[["PETR4.SA", "VALE3.SA"]].dropna()
R = p.pct_change().dropna().loc["2015":"2024"]
X, Y = R["PETR4.SA"], R["VALE3.SA"]

sx, sy = X.std(), Y.std()
rho = X.corr(Y)
w = 0.5                                  # metade em cada ação
var_p = (w**2 * sx**2 + (1 - w)**2 * sy**2
         + 2 * w * (1 - w) * rho * sx * sy)
k = np.sqrt(252)                         # para a base anual
print(f"vol PETR4 {sx * k:.1%}   vol VALE3 {sy * k:.1%}"
      f"   correlação {rho:.3f}")
print(f"vol da carteira pela fórmula: {np.sqrt(var_p) * k:.2%}")

# conferência: a série de retornos da própria carteira
carteira = w * X + (1 - w) * Y
print(f"vol da série da carteira:     {carteira.std() * k:.2%}")
assert np.isclose(var_p, carteira.var())
