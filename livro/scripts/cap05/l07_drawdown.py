import numpy as np
from carteira import Rp

cota = (1 + Rp).cumprod()          # valor de R$ 1 aplicado
hwm = cota.cummax()                # high-water mark
dd = cota / hwm - 1                # drawdown: sempre <= 0

vale = dd.idxmin()
pico = cota[:vale].idxmax()
depois = cota[vale:]
volta = depois[depois >= cota[pico]].index[0]
print(f"drawdown máximo: {dd.min():.1%}")
print(f"pico {pico:%d/%m/%Y}  vale {vale:%d/%m/%Y}  "
      f"recuperação {volta:%d/%m/%Y}")
print(f"queda: {len(cota[pico:vale]) - 1} pregões   "
      f"recuperação: {len(cota[vale:volta]) - 1} pregões")

anos = len(Rp) / 252
cagr = cota.iloc[-1] ** (1 / anos) - 1
print(f"retorno a.a.: {cagr:.1%}   "
      f"Calmar: {cagr / abs(dd.min()):.2f}")
print(f"fração dos dias abaixo do pico: {(dd < 0).mean():.1%}")
