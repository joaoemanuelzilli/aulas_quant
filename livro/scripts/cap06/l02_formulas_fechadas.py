import numpy as np
import pandas as pd

from carteira import NOMES, cdi_anual, insumos, retornos, vol

r = retornos("2015", "2024")
mu, cov = insumos(r)
rf = cdi_anual("2015", "2024")

um = np.ones(len(mu))
x = np.linalg.solve(cov, um)         # resolve Σ x = 1
w_mv = x / x.sum()                   # mínima variância global
y = np.linalg.solve(cov, mu - rf)    # resolve Σ y = μ - r_f
w_tg = y / y.sum()                   # carteira tangente

tabela = pd.DataFrame({"mu": mu, "vol": np.sqrt(np.diag(cov)),
                       "w_mv": w_mv, "w_tg": w_tg},
                      index=NOMES) * 100
print("taxa livre de risco (média diária do CDI x 252): "
      f"{rf:.2%}")
print(tabela.round(1).to_string())
for nome, w in [("mín. variância", w_mv), ("tangente", w_tg)]:
    m, s = w @ mu, vol(w, cov)
    print(f"{nome:15s} retorno {m:7.1%}  vol {s:6.1%}  "
          f"Sharpe {(m - rf) / s:.2f}")
