import numpy as np
from scipy import stats
from carteira import perda

L = perda["2015":"2024"]
nu, m, s = stats.t.fit(L)      # máxima verossimilhança
print(f"nu = {nu:.2f}   m = {m:,.0f}   s = {s:,.0f}")
print(f"desvio-padrão implícito: "
      f"{s * np.sqrt(nu / (nu - 2)):,.0f}"
      f"   amostral: {L.std():,.0f}")

s_mom = L.std() * np.sqrt((nu - 2) / nu)   # escala pelos momentos
for a in (0.95, 0.99):
    q = stats.t.ppf(a, nu)
    print(f"VaR t {a:.0%}: quantil = {q:.3f}  "
          f"ajuste: R$ {m + s * q:,.0f}  "
          f"momentos: R$ {L.mean() + s_mom * q:,.0f}")

errado = m + L.std() * stats.t.ppf(0.99, nu)
print(f"VaR t 99% sem corrigir a escala: R$ {errado:,.0f}")
