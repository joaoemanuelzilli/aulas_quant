import numpy as np
import pandas as pd
from motor import carregar_ohlc, backtest, tabela
from regras import donchian

ohlc, ret, cdi = carregar_ohlc()

ALVO = 0.20                                 # 20% ao ano
vol = ret.rolling(20).std() * np.sqrt(252)  # vol. realizada
tamanho = (ALVO / vol).clip(upper=1.0)      # sem alavancagem

sempre = pd.Series(1.0, index=ohlc.index)
sinal = donchian(ohlc, n=20)
casos = {"PETR4": sempre, "PETR4+alvo": tamanho,
         "Donchian": sinal, "Donch.+alvo": sinal * tamanho}
res = {}
for nome, alvo in casos.items():
    r, giro = backtest(alvo.fillna(0.0), ret, cdi)
    res[nome] = r
    print(f"{nome:12s} giro anual {giro.mean() * 252:4.1f}")
print(tabela(res, cdi))
print(f"tamanho médio: {tamanho.mean():.2f}   "
      f"em março de 2020: {tamanho['2020-03'].mean():.2f}")
