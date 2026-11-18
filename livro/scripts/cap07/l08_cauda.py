import numpy as np
from scipy import stats
from carteira import Rp, cota

r = Rp["2015":"2024"]
print(f"assimetria: {stats.skew(r):.2f}")
print(f"excesso de curtose: {stats.kurtosis(r):.2f}")
q05, q95 = np.quantile(r, [0.05, 0.95])
print(f"razão de cauda q95/|q05|: {q95 / abs(q05):.2f}")
print(f"pior dia: {r.min():.1%} em {r.idxmin():%d/%m/%Y}")

mensal = cota.resample("ME").last().pct_change()["2015":"2024"]
print(f"pior mês: {mensal.min():.1%} em {mensal.idxmin():%m/%Y}")

# quantos desvios-padrão foi o pior dia, e o que a normal diz
k = (r.min() - r.mean()) / r.std()
print(f"pior dia em desvios-padrão: {k:.1f}")
print(f"probabilidade normal de um dia assim: "
      f"{stats.norm.cdf(k):.1e}")
