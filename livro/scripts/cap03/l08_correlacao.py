import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ativos = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"]
p = precos.loc["2015":"2024", ativos].dropna()
r = p.pct_change().dropna()        # retornos simples: é carteira
r.columns = r.columns.str.replace(".SA", "")

print("covariância anualizada:")
print((r.cov() * 252).round(3))
print("correlação de Pearson:")
print(r.corr().round(2))

# a correlação é a covariância dividida pelos desvios
dp = np.sqrt(np.diag(r.cov()))
print(np.allclose(r.cov() / np.outer(dp, dp), r.corr()))

for metodo in ["pearson", "spearman", "kendall"]:
    c = r["PETR4"].corr(r["VALE3"], method=metodo)
    print(f"PETR4 x VALE3, {metodo:8s}: {c:.3f}")

# variância de uma carteira com pesos iguais: w' Sigma w
w = np.full(4, 0.25)
vol_cart = np.sqrt(w @ (r.cov() * 252) @ w)
vol_media = (dp * np.sqrt(252)).mean()
print(f"vol. da carteira: {vol_cart:.1%}"
      f"   média das vols.: {vol_media:.1%}")
