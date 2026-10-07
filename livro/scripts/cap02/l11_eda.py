import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
bcb = pd.read_csv("../../dados/c02_bcb.csv",
                  index_col=0, parse_dates=True)

base = pd.DataFrame({
    "PETR4": precos["PETR4.SA"], "VALE3": precos["VALE3.SA"],
    "ITUB4": precos["ITUB4.SA"], "WEGE3": precos["WEGE3.SA"],
    "Ibovespa": precos["^BVSP"],
    "S&P 500 (US$)": precos["^GSPC"],
    "Dólar": bcb["ptax_venda"],
}).dropna()                         # só datas comuns a todos
base["S&P 500 (R$)"] = base["S&P 500 (US$)"] * base["Dólar"]

ret = base.pct_change().dropna()

base100 = 100 * base / base.iloc[0]
vol = ret.rolling(63).std() * np.sqrt(252)  # só olha para trás
queda = base / base.cummax() - 1            # drawdown
corr = ret.corr()
semanal = base.resample("W-FRI").last().pct_change()
corr_sem = semanal.corr()

if __name__ == "__main__":
    indices = ["Ibovespa", "S&P 500 (US$)", "S&P 500 (R$)",
               "Dólar"]
    print(f"{len(base)} datas, de {base.index[0]:%d/%m/%Y} "
          f"a {base.index[-1]:%d/%m/%Y}\n")
    resumo = pd.DataFrame({
        "base 100 final": base100.iloc[-1].round(0),
        "vol mín": vol.min(), "vol máx": vol.max(),
        "drawdown máx": queda.min(),
        "data do fundo": queda.idxmin().dt.strftime("%d/%m/%Y"),
    }).loc[indices]
    for c in ("vol mín", "vol máx", "drawdown máx"):
        resumo[c] = resumo[c].map(lambda v: f"{v:.1%}")
    print(resumo.to_string())
    print("\ncorrelação com o Ibovespa   diária  semanal")
    for nome in ["PETR4", "S&P 500 (US$)", "S&P 500 (R$)",
                 "Dólar"]:
        print(f"  {nome:<24}{corr.loc['Ibovespa', nome]:>8.2f}"
              f"{corr_sem.loc['Ibovespa', nome]:>9.2f}")
