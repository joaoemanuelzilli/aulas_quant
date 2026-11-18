"""Contango e backwardation no WTI, 1990 a abril de 2024."""
import numpy as np
import pandas as pd
from futuros import ler

f = ler(["cl1", "cl2", "cl3", "cl4"], "1990").dropna()
spread = f["cl4"] - f["cl1"]                 # US$/barril
print(f"pregões: {len(f)}   em contango: {(spread > 0).mean():.1%}")
por_ano = (spread > 0).groupby(spread.index.year).mean()
print("anos com mais de 90% dos dias em contango:",
      list(por_ano[por_ano > 0.9].index))
print("anos com menos de 10%:", list(por_ano[por_ano < 0.1].index))
# inclinação anualizada: o 4º contrato vence ~3 meses depois do 1º
positivo = f["cl1"] > 0                     # 20/04/2020 fica de fora
inclin = np.log(f["cl4"] / f["cl1"][positivo]) / (3 / 12)
tbill = pd.read_csv("../../dados/c09_tbill.csv", index_col=0,
                    parse_dates=True)["tbill"]
for data in ["2008-12-19", "2020-04-21", "2022-03-08"]:
    t = pd.Timestamp(data)
    print(data, f.loc[t].round(2).tolist(),
          f"spread {spread[t]:+.2f}  inclinação {inclin[t]:+.0%}",
          f"juro {tbill[t]:.2f}%")
print("maior spread:", spread.idxmax().date(), spread.max().round(2))
print("menor spread:", spread.idxmin().date(), spread.min().round(2))
