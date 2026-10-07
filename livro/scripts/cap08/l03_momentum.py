import pandas as pd
from motor import carregar, backtest, tabela

ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
         "ABEV3.SA", "BBAS3.SA", "MGLU3.SA"]
p, ret, cdi = carregar(*ACOES, "BOVA11.SA")
bova = ret.pop("BOVA11.SA")
p = p[ACOES]

# preços no último pregão de cada mês
dias = p.index.to_series()
pm = p.loc[dias.groupby(p.index.to_period("M")).max()]
# retorno de 12 meses atrás até 1 mês atrás (pula o último)
sinal = pm.shift(1) / pm.shift(12) - 1
posto = sinal.rank(axis=1, ascending=False)

vencedoras = (posto <= 3) / 3            # as 3 melhores, 1/3 cada
perdedoras = (posto >= 5) / 3            # as 3 piores
casos = {"3 venc.": vencedoras,
         "venc.-perd.": vencedoras - perdedoras,
         "1/7 diário": sinal.notna() / 7}
inicio = sinal.dropna().index[0]
res, giros = {}, {}
for nome, w in casos.items():
    alvo = w.astype(float).reindex(p.index).ffill().fillna(0)
    r, giro = backtest(alvo, ret, cdi, custo=0.001)
    res[nome] = r.loc[inicio:].iloc[1:]
    giros[nome] = f"{giro.loc[inicio:].mean() * 252:.1f}"
# referência sem negociar: 1/7 em cada ação no início, e só
dentro = p.loc[res["3 venc."].index[0]:]
res["mantidas"] = (dentro / dentro.iloc[0]).mean(axis=1) \
    .pct_change().dropna()
print(tabela(res, cdi))
print("giro anual:", " | ".join(giros.values()))
bova = bova.loc[res["3 venc."].index]
aa = (1 + bova).prod() ** (252 / len(bova)) - 1
print(f"{bova.index[0]:%d/%m/%Y} a {bova.index[-1]:%d/%m/%Y}; "
      f"BOVA11: {aa:.1%} a.a.")
for nome in list(res)[:2]:
    m = (1 + res[nome]).resample("ME").prod() - 1
    print(f"assimetria mensal, {nome}: {m.skew():+.2f}")
