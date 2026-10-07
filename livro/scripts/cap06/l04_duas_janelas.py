import pandas as pd

from carteira import (cdi_anual, insumos, maximo_sharpe,
                      minima_variancia, retornos, vol)

est = {}
for jan in ["15-19", "20-24"]:
    ini, fim = "20" + jan[:2], "20" + jan[3:]
    mu, cov = insumos(retornos(ini, fim))
    rf = cdi_anual(ini, fim)
    est[jan] = {"mu": mu, "cov": cov, "rf": rf,
                "MS": maximo_sharpe(mu, cov, rf),
                "MV": minima_variancia(cov)}

t = pd.DataFrame({f"{k} {jan}": est[jan][k] * 100
                  for k in ["mu", "MS", "MV"] for jan in est},
                 index=[a[:5] for a in retornos("2024", "2024")])
print(t.round(1).to_string())
t.to_csv("../../dados/c06_duas_janelas.csv")

a, d = est["15-19"], est["20-24"]     # antes e depois
for k in ["MS", "MV"]:
    w = a[k]
    troca = abs(w - d[k]).sum() / 2
    print(f"\n{k} de 2015-19: {troca:.1%} do patrimônio "
          "trocaria de ativo em 2020-24")
    for nome, e in [("prometido", a), ("realizado", d)]:
        m, s = w @ e["mu"], vol(w, e["cov"])
        print(f"  {nome}: retorno {m:6.1%}  vol {s:5.1%}  "
              f"Sharpe {(m - e['rf']) / s:5.2f}")
