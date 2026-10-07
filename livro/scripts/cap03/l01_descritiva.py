import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ativos = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
          "BOVA11.SA"]
p = precos.loc["2015":"2024", ativos].dropna()
r = np.log(p / p.shift(1)).dropna() * 100    # em % ao dia

q = r.quantile([0.05, 0.25, 0.75, 0.95]).T   # uma linha por ativo
iqr = q[0.75] - q[0.25]
resumo = pd.DataFrame({
    "média": r.mean(), "mediana": r.median(),
    "desvio": r.std(), "IQR": iqr, "IQR/1,349": iqr / 1.349,
    "q5%": q[0.05], "q95%": q[0.95],
    "mín": r.min(), "máx": r.max(),
})
resumo.index = resumo.index.str.replace(".SA", "")
print(f"{len(r)} retornos diários por ativo, em %")
print(resumo.round(2).to_string())
