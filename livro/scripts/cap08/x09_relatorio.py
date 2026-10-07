"""Números do relatório-modelo (Donchian em PETR4, walk-forward).

Grava dados/c08_relatorio.csv com as curvas usadas na figura.
"""
import numpy as np
import pandas as pd
from motor import backtest, metricas, operacoes
from l07_walkforward import (walk_forward, rets, alvos, ret, cdi, sharpe, TREINO, GRADE)

wf, tab, costura = walk_forward()
ini, fim = wf.index[0], wf.index[-1]
c = cdi.loc[ini:]
n_otimo = rets.iloc[TREINO:].apply(sharpe).idxmax()
casos = {
    "na amostra (n=%d)" % n_otimo: (rets[n_otimo].loc[ini:], alvos[n_otimo]),
    "fora (walk-forward)": (wf, costura),
    "PETR4 comprada": (ret.loc[ini:], pd.Series(1.0, index=ret.index)),
}
print(f"período: {ini:%d/%m/%Y} a {fim:%d/%m/%Y} ({len(wf)} pregões); candidatos: {len(GRADE)}")
linhas = {}
for nome, (r, alvo) in casos.items():
    m = metricas(r, c)
    a = alvo.loc[ini:]
    ops = operacoes(a, ret.loc[ini:], c)
    giro = a.shift(1).fillna(0).diff().abs().sum() / len(a) * 252
    m["operações"] = len(ops)
    m["taxa de acerto"] = (ops > 0).mean()
    m["giro anual"] = giro
    m["tempo comprado"] = a.mean()
    m["assimetria mensal"] = ((1 + r).resample("ME").prod() - 1).skew()
    linhas[nome] = m
t = pd.DataFrame(linhas)
pd.set_option("display.width", 200)
print(t.round(3))
print(f"CDI a.a. no período: {(1 + c).prod() ** (252 / len(c)) - 1:.1%}")
dd = (1 + wf).cumprod() / (1 + wf).cumprod().cummax().clip(lower=1.0) - 1
print("fundo do maior drawdown do WF:", dd.idxmin().date(), f"{dd.min():.1%}")
fundo = dd < 0
blocos = fundo.groupby((~fundo).cumsum())
maior = blocos.sum().idxmax()
trecho = dd[(~fundo).cumsum() == maior]
print("drawdown mais longo do WF:", trecho.index[0].date(), "a", trecho.index[-1].date(), len(trecho) - 1, "pregões")
print("em drawdown no último pregão:", bool(dd.iloc[-1] < 0), f"{dd.iloc[-1]:.1%}", "desde", dd[dd == 0].index[-1].date() if (dd == 0).any() else None)
print(f"valor final de R$ 1: WF {(1 + wf).prod():.2f}  PETR4 {(1 + ret.loc[ini:]).prod():.2f}  CDI {(1 + c).prod():.2f}")
pd.DataFrame({"wf": wf, "petr4": ret.loc[ini:], "cdi": c, "fixo": rets[n_otimo].loc[ini:]}).to_csv("../../dados/c08_relatorio.csv")
