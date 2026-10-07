import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/bruto/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
bcb = pd.read_csv("../../dados/c02_bcb.csv",
                  index_col=0, parse_dates=True)

yahoo = precos["BRL=X"].dropna()
d = pd.DataFrame({"Ibovespa": precos["^BVSP"],
                  "BRL=X": yahoo,
                  "BRL=X seguinte": yahoo.shift(-1),
                  "PTAX": bcb["ptax_venda"]}).dropna()
print(d.loc["2017-05-17":"2017-05-22"])

r = d.pct_change()
print("\ncorrelação com o retorno diário do Ibovespa")
for nome in ["BRL=X", "BRL=X seguinte", "PTAX"]:
    print(f"  {nome:<16}{r['Ibovespa'].corr(r[nome]):+.3f}")

# depois de corrigir a data, as duas fontes concordam?
desvio = d["BRL=X seguinte"] / d["PTAX"] - 1
print(f"\ndias com desvio acima de 3%: "
      f"{(desvio.abs() > 0.03).sum()} de {len(d)}")
vol_2010 = r.loc["2010"].std() * np.sqrt(252)
print(f"volatilidade em 2010:  BRL=X {vol_2010['BRL=X']:.1%}"
      f"   PTAX {vol_2010['PTAX']:.1%}")
