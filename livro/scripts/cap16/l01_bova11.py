import numpy as np
import pandas as pd
from algoritmos import drawdown_maximo, melhor_negocio

px = pd.read_csv("../../dados/precos_ajustados.csv", index_col=0,
                 parse_dates=True)["BOVA11.SA"].dropna()
d = px.index
print(f"{len(px)} pregões, {d[0]:%d/%m/%Y} a {d[-1]:%d/%m/%Y}")

# em log-preços, o maior p[j] - p[i] é o maior retorno
lucro, i, j = melhor_negocio(list(np.log(px)))
print(f"compra {d[i]:%d/%m/%Y} a {px.iloc[i]:.2f}, "
      f"venda {d[j]:%d/%m/%Y} a {px.iloc[j]:.2f}: "
      f"{np.exp(lucro) - 1:.1%}")

dd = drawdown_maximo(list(px))
assert np.isclose(dd, (px / px.cummax() - 1).min())
print(f"drawdown máximo: {dd:.1%}")
