import numpy as np
from carteira import cota

hwm = cota.cummax()                # high-water mark
dd = cota / hwm - 1                # drawdown: sempre <= 0

vale = dd.idxmin()
pico = cota[:vale].idxmax()
depois = cota[vale:]
recuperou = depois[depois >= cota[pico]]
queda = len(cota[pico:vale]) - 1
print(f"drawdown máximo: {dd.min():.1%}")
print(f"pico {pico:%d/%m/%Y}  vale {vale:%d/%m/%Y}  "
      f"queda: {queda} pregões")
if len(recuperou):                 # a cota já voltou ao pico?
    volta = recuperou.index[0]
    recuperacao = len(cota[vale:volta]) - 1
    print(f"recuperação {volta:%d/%m/%Y}: {recuperacao} pregões"
          f"   tempo submerso: {queda + recuperacao}")
else:
    print("a cota ainda não voltou ao pico")

anos = len(cota) / 252
cagr = cota.iloc[-1] ** (1 / anos) - 1
print(f"retorno a.a.: {cagr:.1%}   "
      f"Calmar: {cagr / abs(dd.min()):.2f}")
print(f"fração dos dias abaixo do pico: {(dd < 0).mean():.1%}")
