import numpy as np
from scipy import stats
from carteira import perda

L = perda["2015":"2024"]
mu, sig = L.mean(), L.std()
nu, m, s = stats.t.fit(L)

print("nível   método      VaR       ES    ES/VaR")
for a in (0.95, 0.975, 0.99):
    z, q = stats.norm.ppf(a), stats.t.ppf(a, nu)
    v_h = np.quantile(L, a)
    linhas = {
        # média das perdas iguais ou maiores que o VaR
        "histórico": (v_h, L[L >= v_h].mean()),
        "normal": (mu + sig * z,
                   mu + sig * stats.norm.pdf(z) / (1 - a)),
        "t": (m + s * q,
              m + s * stats.t.pdf(q, nu) / (1 - a)
              * (nu + q ** 2) / (nu - 1)),
    }
    for nome, (v, es) in linhas.items():
        print(f"{a:6.1%}  {nome:9s} {v:8,.0f} {es:8,.0f}"
              f"   {es / v:.2f}")
