"""Quatro contas de apoio à leitura do backtest: erro-padrão do
Sharpe, mistura passiva de mesma exposição, placebo (o sinal
do modelo deslocado no tempo, em todos os deslocamentos
circulares possíveis: mesma exposição em cada ação, giro
parecido, nenhuma informação sobre os dias) e a carteira
parada nos pesos médios que o modelo deu a cada ação."""
import numpy as np
import pandas as pd
from l08_backtest import (carteira, cdi, pesos, resultado, ret,
                          todas)

liq, bruto, giro = carteira(pesos)
exc = liq - cdi
anos = len(liq) / 252
sr = exc.mean() / exc.std() * np.sqrt(252)
print(f"Sharpe {sr:.2f} | anos {anos:.2f} | erro-padrão "
      f"{1 / np.sqrt(anos):.2f} | t {sr * np.sqrt(anos):.2f}")
custo = (bruto - liq).mean() * 252
eb = bruto - cdi
print(f"custo: {custo:.3f} a.a. = {custo / (exc.std() * np.sqrt(252)):.2f}"
      f" de Sharpe | Sharpe bruto "
      f"{eb.mean() / eb.std() * np.sqrt(252):.2f}")

expo = pesos.sum(axis=1).mean()
mist = resultado(todas * expo)
print(f"mistura fixa, {expo:.1%} em ações: retorno "
      f"{mist['retorno a.a.']:.3f} | vol {mist['vol a.a.']:.3f} | "
      f"Sharpe {mist['Sharpe']:.2f} | drawdown "
      f"{mist['drawdown máx.']:.3f} | giro {mist['giro a.a.']:.1f}")

T = len(pesos)
linhas = []
for k in range(1, T):                  # todos os deslocamentos
    p = pd.DataFrame(np.roll(pesos.to_numpy(), k, axis=0),
                     index=pesos.index, columns=pesos.columns)
    r = resultado(p)
    linhas.append((r["Sharpe"], r["retorno a.a."], r["giro a.a."]))
pl = pd.DataFrame(linhas, columns=["Sharpe", "retorno", "giro"])
print(f"placebo ({len(pl)} deslocamentos): Sharpe médio "
      f"{pl['Sharpe'].mean():.2f}, dp {pl['Sharpe'].std():.2f} | "
      f"retorno médio {pl['retorno'].mean():.3f} | giro médio "
      f"{pl['giro'].mean():.1f}")
print(f"fração dos placebos com Sharpe abaixo do modelo: "
      f"{(pl['Sharpe'] < sr).mean():.2f}")

# o deslocamento preserva a fração de dias em que cada ação fica
# na carteira; uma carteira parada nesses pesos médios, sem
# previsão diária, mostra quanto do resultado vem só daí
frac = (pesos > 0).mean()
aa = (1 + ret).prod() ** (252 / len(ret)) - 1
print("dias na carteira, por ação:", frac.round(2).to_dict())
print("retorno a.a. no período, por ação:", aa.round(3).to_dict())
fixa = resultado(todas * frac)
print(f"pesos médios fixos: retorno {fixa['retorno a.a.']:.3f} | "
      f"sem custos {fixa['sem custos']:.3f} | Sharpe "
      f"{fixa['Sharpe']:.2f} | giro {fixa['giro a.a.']:.1f}")
