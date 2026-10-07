import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
bcb = pd.read_csv("../../dados/c02_bcb.csv",
                  index_col=0, parse_dates=True)

t = pd.DataFrame({"sp_usd": precos["^GSPC"],
                  "dolar": bcb["ptax_venda"],
                  "ibov": precos["^BVSP"]}).dropna()
t["sp_brl"] = t["sp_usd"] * t["dolar"]    # pontos x (R$ por US$)

total = t.iloc[-1] / t.iloc[0] - 1
print(f"{t.index[0]:%d/%m/%Y} a {t.index[-1]:%d/%m/%Y}")
print(f"S&P 500 em dólares: {total['sp_usd']:+8.1%}")
print(f"dólar em reais:     {total['dolar']:+8.1%}")
print(f"S&P 500 em reais:   {total['sp_brl']:+8.1%}")
composto = (1 + total["sp_usd"]) * (1 + total["dolar"]) - 1
print(f"(1+R_usd)(1+R_fx)-1 {composto:+8.1%}")

r = t.pct_change().dropna()
vol = r.std() * np.sqrt(252)
print("\nvolatilidade anualizada")
print(vol.map(lambda v: f"{v:.1%}").to_string())
print("\ncorrelação dos retornos diários")
print(r.corr().round(2).to_string())
