import numpy as np
from carteira import cota, pregoes

hwm = cota.cummax()                # high-water mark
dd = cota / hwm - 1                # drawdown: sempre <= 0

vale = dd.idxmin()
pico = cota[:vale].idxmax()
depois = cota[vale:]
volta = depois[depois >= cota[pico]].index[0]
queda = len(cota[pico:vale]) - 1
recuperacao = len(cota[vale:volta]) - 1
print(f"drawdown máximo: {dd.min():.1%}")
print(f"pico {pico:%d/%m/%Y}  vale {vale:%d/%m/%Y}  "
      f"recuperação {volta:%d/%m/%Y}")
print(f"queda: {queda} pregões   recuperação: {recuperacao}"
      f"   tempo submerso: {queda + recuperacao}")

anos = len(pregoes[cota.index[0]:]) / 252   # pregões da B3
cagr = cota.iloc[-1] ** (1 / anos) - 1
print(f"retorno a.a.: {cagr:.1%}   "
      f"Calmar: {cagr / abs(dd.min()):.2f}")
print(f"fração dos dias abaixo do pico: {(dd < 0).mean():.1%}")
