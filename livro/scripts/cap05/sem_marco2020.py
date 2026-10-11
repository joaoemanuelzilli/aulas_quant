"""Quanto das medidas de cauda depende de março de 2020."""
import numpy as np
from scipy import stats
from carteira import perda, Rp

L = perda["2015":"2024"]
print("cinco maiores perdas:")
print(L.nlargest(5).round(0).to_string())


def resumo(x, nome):
    linha = f"{nome:22s} {len(x)} dias"
    for a in (0.95, 0.99):
        v = np.quantile(x, a)
        es = x[x >= v].mean()
        linha += f" | {a:.0%}: VaR {v:,.0f} ES {es:,.0f} {es / v:.2f}"
    print(linha, f"| nu = {stats.t.fit(x)[0]:.2f}")


sem_marco = L[(L.index < "2020-03-01") | (L.index > "2020-03-31")]
resumo(L, "amostra completa")
resumo(sem_marco, "sem março de 2020")
resumo(L[L.index.year != 2020], "sem 2020")
resumo(L["2015":"2019"], "2015 a 2019")
resumo(L["2020":"2024"], "2020 a 2024")
print("pregões de março de 2020:", len(L) - len(sem_marco))
r = Rp["2015":"2024"]
print(f"autocorrelação de ordem 1: {r.autocorr(1):.3f}; "
      f"sem 2020: {r[r.index.year != 2020].autocorr(1):.3f}")
