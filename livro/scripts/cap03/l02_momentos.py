import numpy as np
import pandas as pd
from scipy import stats

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()    # 2010 a 2025

print("n =", len(r))
print(f"assimetria  pandas {r.skew():.4f}"
      f"   scipy {stats.skew(r):.4f}")
k_pearson = stats.kurtosis(r, fisher=False)
print(f"curtose     pandas {r.kurt():.4f}"
      f"   scipy {stats.kurtosis(r):.4f}")
print(f"            scipy com fisher=False {k_pearson:.4f}")

# a mesma série em semanas e meses (log-retornos somam no tempo)
for nome, regra in [("semanal", "W"), ("mensal", "ME")]:
    # min_count=1: período sem pregão fica ausente, e não zero
    ag = r.resample(regra).sum(min_count=1).dropna()
    print(f"{nome:8s} n = {len(ag):4d}"
          f"   assimetria {ag.skew():5.2f}"
          f"   excesso de curtose {ag.kurt():5.2f}")
