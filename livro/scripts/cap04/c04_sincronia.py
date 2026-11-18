"""O câmbio do arquivo congelado está defasado um dia em relação à B3
(números da armadilha sobre séries dessincronizadas)."""
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
p = precos[["^BVSP", "BRL=X"]].dropna()
r = p.pct_change().loc["2015":"2024"]
print(len(r), "dias")
for d in [-2, -1, 0, 1, 2]:
    c = r["^BVSP"].corr(r["BRL=X"].shift(d))
    print(f"Ibovespa em t, dólar em t{-d:+d}: {c:+.3f}")
