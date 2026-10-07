"""Momentum transversal: sensibilidade à presença de MGLU3 e ao
número de papéis comprados (números citados no texto)."""
import pandas as pd
from motor import carregar, backtest, metricas

TODAS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "ABEV3.SA", "BBAS3.SA", "MGLU3.SA"]


def rodar(acoes, k):
    p, ret, cdi = carregar(*TODAS, "BOVA11.SA")   # mesmas datas do l03_momentum.py
    p, ret = p[acoes], ret[acoes]
    dias = p.index.to_series()
    pm = p.loc[dias.groupby(p.index.to_period("M")).max()]
    sinal = pm.shift(1) / pm.shift(12) - 1
    posto = sinal.rank(axis=1, ascending=False)
    inicio = sinal.dropna().index[0]
    out = {}
    for nome, w in {"vencedoras": (posto <= k) / k, "todas": sinal.notna() / len(acoes)}.items():
        alvo = w.astype(float).reindex(p.index).ffill().fillna(0)
        r, _ = backtest(alvo, ret, cdi, custo=0.001)
        r = r.loc[inicio:].iloc[1:]
        out[nome] = metricas(r, cdi.loc[r.index])
    return out


for rot, acoes in [("7 ações", TODAS), ("sem MGLU3", TODAS[:-1])]:
    for k in (2, 3):
        m = rodar(acoes, k)
        print(f"{rot:10s} k={k}: vencedoras {m['vencedoras']['retorno a.a.']:.1%} a.a., Sharpe {m['vencedoras']['Sharpe']:.2f} | "
              f"pesos iguais {m['todas']['retorno a.a.']:.1%} a.a., Sharpe {m['todas']['Sharpe']:.2f}")

# MGLU3: quanto subiu e quanto tempo ficou entre as três vencedoras
p, ret, cdi = carregar(*TODAS, "BOVA11.SA")
p = p[TODAS]
dias = p.index.to_series()
pm = p.loc[dias.groupby(p.index.to_period("M")).max()]
posto = (pm.shift(1) / pm.shift(12) - 1).rank(axis=1, ascending=False)
m = p["MGLU3.SA"]
print(f"MGLU3: preço ajustado multiplicado por {m['2020'].iloc[-1] / m['2015'].iloc[-1]:.0f} do fim de 2015 ao fim de 2020; "
      f"queda de {m.iloc[-1] / m['2020'].iloc[-1] - 1:.0%} do fim de 2020 ao fim de 2025")
top = (posto["MGLU3.SA"] <= 3)
print(f"meses entre as 3 vencedoras: {top['2016-07':'2021-06'].mean():.0%} de jul/2016 a jun/2021; {top[posto['MGLU3.SA'].notna()].mean():.0%} no período todo")

# giro que viria só da troca mensal de papéis (sem a deriva diária dos pesos)
ini = (pm.shift(1) / pm.shift(12) - 1).dropna().index[0]
for nome, w in {"3 vencedoras": (posto <= 3) / 3, "vencedoras - perdedoras": (posto <= 3) / 3 - (posto >= 5) / 3}.items():
    a = w.astype(float).reindex(p.index).ffill().fillna(0)
    g = a.shift(1).fillna(0).diff().abs().sum(axis=1)
    print(f"giro anual só das trocas mensais, {nome}: {g.loc[ini:].mean() * 252:.1f}")
