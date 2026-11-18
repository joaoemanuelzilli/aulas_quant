import numpy as np
import pandas as pd
from motor import carregar, backtest

p, ret, cdi = carregar("BOVA11.SA")
p, ret = p["BOVA11.SA"], ret["BOVA11.SA"]

# teste 1: comprado o tempo todo, sem custo = o próprio ativo
sempre = pd.Series(1.0, index=p.index)
r, giro = backtest(sempre, ret, cdi, custo=0)
print(f"motor: {(1 + r.iloc[1:]).prod():.6f}   "
      f"preço final / inicial: {p.iloc[-1] / p.iloc[0]:.6f}")

# teste 2: uma regra que "decide" olhando o retorno do dia
vidente = (ret > 0).astype(float)
for d in (0, 1):
    r, giro = backtest(vidente, ret, cdi, custo=0, defasagem=d)
    aa = (1 + r).prod() ** (252 / len(r)) - 1
    print(f"defasagem de {d} dia(s): {aa:8.1%} ao ano")

# teste 3: o custo cobrado é custo x giro
r0, giro = backtest(vidente, ret, cdi, custo=0)
r1, _ = backtest(vidente, ret, cdi, custo=0.001)
print(f"giro anual: {giro.sum() / len(giro) * 252:.0f} vezes o "
      f"capital   custo anual: {(r0 - r1).mean() * 252:.1%}")
