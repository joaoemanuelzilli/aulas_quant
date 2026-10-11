import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
petr = precos["PETR4.SA"].dropna()

simples = petr.pct_change().dropna()         # R_t = P_t/P_{t-1}-1
log = np.log(petr / petr.shift(1)).dropna()  # r_t = ln(1 + R_t)

ano = "2024"
composto = (1 + simples.loc[ano]).prod() - 1  # compõe os simples
via_log = np.exp(log.loc[ano].sum()) - 1      # soma os logs
soma = simples.loc[ano].sum()                 # o erro clássico
dif = (simples - log).abs().max()

# conferência independente: só os preços das duas pontas
p_ini = petr.loc[:"2023"].iloc[-1]   # último fechamento de 2023
p_fim = petr.loc[ano].iloc[-1]       # último fechamento de 2024
assert np.isclose(composto, p_fim / p_ini - 1)
assert np.isclose(composto, via_log)

print(f"retorno de {ano}, compondo os simples: {composto:.4%}")
print(f"retorno de {ano}, somando os logs:     {via_log:.4%}")
print(f"soma ingênua dos retornos simples:      {soma:.4%}")
print(f"maior diferença diária |R - r|:         {dif:.4%}")
