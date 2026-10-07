import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
petr = precos["PETR4.SA"].dropna()

simples = petr.pct_change().dropna()         # R_t = P_t/P_{t-1}-1
log = np.log(petr / petr.shift(1)).dropna()  # r_t = ln(1 + R_t)

ano = "2024"
composto = (1 + simples[ano]).prod() - 1     # compõe os simples
via_log = np.exp(log[ano].sum()) - 1         # soma os logs
soma = simples[ano].sum()                    # o erro clássico
dif = (simples - log).abs().max()

print(f"retorno de {ano}, compondo os simples: {composto:.4%}")
print(f"retorno de {ano}, somando os logs:     {via_log:.4%}")
print(f"soma ingênua dos retornos simples:      {soma:.4%}")
print(f"maior diferença diária |R - r|:         {dif:.4%}")
