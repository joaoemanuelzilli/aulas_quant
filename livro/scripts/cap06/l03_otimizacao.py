import numpy as np
import pandas as pd

from carteira import (cdi_anual, insumos, maximo_sharpe,
                      minima_variancia, retornos, vol)

r = retornos("2015", "2024")
mu, cov = insumos(r)
rf = cdi_anual("2015", "2024")
n = len(mu)

carteiras = {"1/N": np.full(n, 1 / n),
             "mín. var.": minima_variancia(cov),
             "máx. Sharpe": maximo_sharpe(mu, cov, rf)}
tabela = pd.DataFrame(carteiras, index=r.columns.str[:5])
print((tabela * 100).round(1).to_string())
for nome, w in carteiras.items():
    m, s = w @ mu, vol(w, cov)
    print(f"{nome:12s} retorno {m:6.1%}  vol {s:6.1%}  "
          f"Sharpe {(m - rf) / s:.2f}")
