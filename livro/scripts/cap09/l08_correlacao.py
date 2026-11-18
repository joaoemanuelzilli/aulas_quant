"""WTI, Brent e gás: níveis, retornos e calendário."""
import numpy as np
from futuros import ler

p = ler(["wti", "brent", "hh"], "2008")
so_um = p[["wti", "brent"]].notna().sum(axis=1) == 1
print(f"datas com só WTI ou só Brent: {so_um.sum()} de {len(p)}")
p = p.dropna()
p = p[(p > 0).all(axis=1)]            # tira 20/04/2020 (WTI < 0)
lp = np.log(p)
print("correlação dos log-preços com o WTI:",
      lp.corr().loc["wti", ["brent", "hh"]].round(2).tolist())
for nome, regra in [("diário", None), ("semanal", "W-FRI"),
                    ("mensal", "ME")]:
    x = lp if regra is None else lp.resample(regra).last()
    r = x.diff().dropna()
    print(f"retorno {nome:7s} WTI x Brent "
          f"{r.wti.corr(r.brent):.2f}  WTI x gás "
          f"{r.wti.corr(r.hh):.2f}  ({len(r)} obs.)")
r = lp.diff().dropna()
ontem = r.shift()
print(f"Brent hoje x WTI ontem {r.brent.corr(ontem.wti):.3f};",
      f"WTI hoje x Brent ontem {r.wti.corr(ontem.brent):.3f}")
