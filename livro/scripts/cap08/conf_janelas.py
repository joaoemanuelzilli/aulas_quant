"""Instabilidade dos pesos em pares de janelas vizinhas
(números citados na Seção 6.7)."""
import numpy as np

from carteira import (cdi_anual, insumos, maximo_sharpe,
                      minima_variancia, retornos, vol)

PARES = [(("2012", "2014"), ("2015", "2017")),
         (("2015", "2017"), ("2018", "2020")),
         (("2018", "2020"), ("2021", "2023")),
         (("2021", "2023"), ("2024", "2025")),
         (("2012", "2016"), ("2017", "2021")),
         (("2013", "2017"), ("2018", "2022")),
         (("2015", "2019"), ("2020", "2024"))]


def estimar(ini, fim):
    mu, cov = insumos(retornos(ini, fim))
    rf = cdi_anual(ini, fim)
    return {"mu": mu, "cov": cov, "rf": rf,
            "MS": maximo_sharpe(mu, cov, rf),
            "MV": minima_variancia(cov)}


def sharpe(w, e):
    return (w @ e["mu"] - e["rf"]) / vol(w, e["cov"])


abaixo = acima_1n = 0
for antes, depois in PARES:
    a, d = estimar(*antes), estimar(*depois)
    troca = {k: abs(a[k] - d[k]).sum() / 2 for k in ["MS", "MV"]}
    prom, real = sharpe(a["MS"], a), sharpe(a["MS"], d)
    um_n = sharpe(np.full(7, 1 / 7), d)
    abaixo += real < prom
    acima_1n += real > um_n
    vol_p, vol_r = vol(a["MV"], a["cov"]), vol(a["MV"], d["cov"])
    print(f"{antes[0]}-{antes[1][2:]} -> {depois[0]}-"
          f"{depois[1][2:]}: troca MS {troca['MS']:4.0%} "
          f"MV {troca['MV']:4.0%} | Sharpe MS prometido "
          f"{prom:5.2f} realizado {real:5.2f} 1/N {um_n:5.2f} | "
          f"vol MV {vol_p:.1%} -> {vol_r:.1%}")
print(f"realizado abaixo do prometido em {abaixo} de "
      f"{len(PARES)} pares; acima de 1/N em {acima_1n}")

# médias e volatilidades nas duas janelas de cinco anos
a, d = estimar("2015", "2019"), estimar("2020", "2024")
va, vd = np.sqrt(np.diag(a["cov"])), np.sqrt(np.diag(d["cov"]))
for rot, s in [("sete ações", slice(None)),
               ("sem MGLU3", slice(6))]:
    print(rot, "correlação entre janelas: médias",
          round(np.corrcoef(a["mu"][s], d["mu"][s])[0, 1], 2),
          "volatilidades",
          round(np.corrcoef(va[s], vd[s])[0, 1], 2))
