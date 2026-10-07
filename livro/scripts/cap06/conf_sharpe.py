"""Conferência: a diferença de Sharpe entre máx. Sharpe e 1/N
é distinguível de zero? Bootstrap em blocos de 21 pregões,
pareado (as duas séries são reamostradas juntas)."""
import numpy as np
import pandas as pd

from experimento import carregar_cdi

d = pd.read_csv("../../dados/c06_fora_da_amostra.csv",
                index_col=0, parse_dates=True)
cdi = carregar_cdi(d.index)
ex = d.sub(cdi, axis=0)[["máx. Sharpe", "1/N"]].values


def dif_sharpe(x):
    s = x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(252)
    return s[0] - s[1]


rng = np.random.default_rng(6)
bloco, t = 21, len(ex)
difs = []
for _ in range(5000):
    ini = rng.integers(0, t - bloco + 1, size=t // bloco + 1)
    idx = (ini[:, None] + np.arange(bloco)).ravel()[:t]
    difs.append(dif_sharpe(ex[idx]))
lo, hi = np.percentile(difs, [2.5, 97.5])
print(f"diferença de Sharpe: {dif_sharpe(ex):.2f}")
print(f"intervalo de 95% (bootstrap): [{lo:.2f}, {hi:.2f}]")
print(f"fração de reamostras com diferença < 0: "
      f"{np.mean(np.array(difs) < 0):.1%}")
for a, b in [("2015", "2019"), ("2020", "2025")]:
    e = d.sub(cdi, axis=0).loc[a:b]
    s = e.mean() / e.std() * np.sqrt(252)
    print(a, b, s.round(2).to_dict())

# hipótese de execução: um pregão de atraso entre o cálculo
# dos pesos e a montagem da carteira
from carteira import ACOES, retornos
from experimento import resumo, simular

r = retornos("2012", "2025")
cdi_r = carregar_cdi(r.index)
for nome, cols in [("sete ações", ACOES),
                   ("sem MGLU3", [a for a in ACOES
                                  if a != "MGLU3.SA"])]:
    d1, g1 = simular(r[cols], cdi_r, atraso=1)
    t = resumo(d1, g1, cdi_r)["Sharpe"].round(2)
    print("atraso de 1 pregão,", nome, t.to_dict())
