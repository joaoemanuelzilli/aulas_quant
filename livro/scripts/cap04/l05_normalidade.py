import numpy as np
import pandas as pd
from scipy import stats

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()

def jarque_bera(x):
    n = len(x)
    s = stats.skew(x)                  # assimetria amostral
    k = stats.kurtosis(x)              # excesso de curtose
    jb = n / 6 * (s ** 2 + k ** 2 / 4)
    return jb, stats.chi2.sf(jb, df=2)

mensal = r.resample("ME").sum(min_count=1).dropna()
anual = r.resample("YE").sum(min_count=1).dropna()
series = {"diário": r, "mensal": mensal, "anual": anual}
for nome, x in series.items():
    jb, p = jarque_bera(x)
    sw = stats.shapiro(x)
    print(f"{nome:7s} n = {len(x):4d}   JB = {jb:8.1f}"
          f"  p = {p:8.2g}   Shapiro-Wilk: p = {sw.pvalue:.2g}")

print(f"scipy, diário: JB = {stats.jarque_bera(r).statistic:.1f}")
