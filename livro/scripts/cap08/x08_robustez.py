"""Testes de robustez da regra de Donchian com walk-forward.

Não é impresso no livro; gera os números da seção de robustez
e grava dados/c08_robustez.csv para a figura.
"""
import numpy as np
import pandas as pd
from motor import carregar, carregar_ohlc, backtest, metricas
from regras import donchian

GRADE = list(range(10, 105, 5))
TREINO, TESTE = 756, 252
ATIVOS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "ABEV3.SA", "BBAS3.SA", "MGLU3.SA", "BOVA11.SA"]


def sharpe(r, cdi):
    ex = r - cdi.loc[r.index]
    return ex.mean() / ex.std() * np.sqrt(252)


def walk_forward(ohlc, ret, cdi, custo=0.001, atraso=1, grade=GRADE, TREINO=TREINO, TESTE=TESTE):
    """Mesmo procedimento do l07 (janela rolante), para qualquer ativo."""
    alvos = pd.DataFrame({n: donchian(ohlc, n) for n in grade})
    rets = alvos.apply(lambda a: backtest(a, ret, cdi, custo, atraso)[0])
    costura = pd.Series(0.0, index=ret.index)
    for ini in range(TREINO, len(ret), TESTE):
        treino = rets.iloc[ini - TREINO:ini]
        n = treino.apply(sharpe, cdi=cdi).idxmax()
        fim = min(ini + TESTE, len(ret))
        costura.iloc[ini:fim] = alvos[n].iloc[ini:fim]
    r, giro = backtest(costura, ret, cdi, custo, atraso)
    return r.iloc[TREINO:], rets, costura


ohlc, ret, cdi = carregar_ohlc()

# 1. parâmetro: Sharpe de cada n fixo, amostra inteira e período fora da amostra
wf, rets, _ = walk_forward(ohlc, ret, cdi)
sr_tudo = rets.apply(sharpe, cdi=cdi)
sr_fora = rets.iloc[TREINO:].apply(sharpe, cdi=cdi)
print("1. Sharpe por n (amostra inteira):", sr_tudo.round(2).to_dict())
print("   Sharpe por n (2013-2025):     ", sr_fora.round(2).to_dict())
print(f"   n com Sharpe > 0,30 na amostra inteira: {[n for n in GRADE if sr_tudo[n] > 0.30]}")
print(f"   walk-forward: {sharpe(wf, cdi):.2f}")

# 2. custos e atraso de execução
print("2. Sharpe do walk-forward por custo e atraso")
linhas = []
for custo in (0, 0.0005, 0.001, 0.002, 0.005):
    s1 = sharpe(walk_forward(ohlc, ret, cdi, custo, 1)[0], cdi)
    s2 = sharpe(walk_forward(ohlc, ret, cdi, custo, 2)[0], cdi)
    linhas.append((custo, s1, s2))
    print(f"   custo {custo:.2%}: atraso 1 -> {s1:.2f}   atraso 2 -> {s2:.2f}")

# 3. subperíodos (walk-forward de l07)
print("3. subperíodos do walk-forward")
comprado = ret.loc[wf.index]
for a, b in [("2013", "2016"), ("2017", "2020"), ("2021", "2025")]:
    m = metricas(wf[a:b], cdi[a:b])
    print(f"   {a}-{b}: retorno {m['retorno a.a.']:.1%} a.a., Sharpe {m['Sharpe']:.2f}, "
          f"drawdown {m['drawdown máx.']:.1%} | PETR4 comprada: Sharpe {sharpe(comprado[a:b], cdi):.2f}")
anual = (1 + wf).groupby(wf.index.year).prod() - 1
cdi_anual = (1 + cdi.loc[wf.index]).groupby(wf.index.year).prod() - 1
print("   anos com retorno acima do CDI:", int((anual > cdi_anual).sum()), "de", len(anual))
print("   retorno por ano:", (anual * 100).round(0).astype(int).to_dict())
sem2016 = wf[wf.index.year != 2016]
print(f"   Sharpe sem o ano de 2016: {sharpe(sem2016, cdi):.2f}")

# 4. outros ativos: só há fechamento, então o canal usa fechamentos
print("4. mesmo procedimento em outros ativos (canal de fechamentos)")
saida = []
for t in ATIVOS:
    pt, rt, ct = carregar(t)
    rt = rt[t]
    falso = pd.DataFrame({"High": pt[t], "Low": pt[t], "Close": pt[t]})
    wft, retst, _ = walk_forward(falso, rt, ct)
    bh = rt.loc[wft.index]
    fixo20 = retst[20].loc[wft.index]
    saida.append({"ativo": t.replace(".SA", ""), "wf": sharpe(wft, ct), "n20": sharpe(fixo20, ct), "comprado": sharpe(bh, ct),
                  "inicio": wft.index[0].date()})
    print(f"   {t:10s} desde {wft.index[0].date()}: WF {saida[-1]['wf']:+.2f}   n=20 fixo {saida[-1]['n20']:+.2f}   "
          f"comprado {saida[-1]['comprado']:+.2f}")
df = pd.DataFrame(saida)
print(f"   WF supera o ativo comprado em {(df.wf > df.comprado).sum()} de {len(df)}; WF com Sharpe positivo em {(df.wf > 0).sum()}")
print(f"   médias: WF {df.wf.mean():.2f}, comprado {df.comprado.mean():.2f}")
df.to_csv("../../dados/c08_robustez_ativos.csv", index=False)
pd.DataFrame({"n": GRADE, "sr_tudo": sr_tudo.values, "sr_fora": sr_fora.values}).to_csv("../../dados/c08_robustez_n.csv", index=False)

# 5. tamanho das janelas de treino e teste (PETR4, canal com máximas e mínimas)
print("5. Sharpe do walk-forward por tamanho de janela (treino, teste em pregões)")
vals = []
for tr in (504, 756, 1008, 1260):
    for te in (126, 252):
        r = walk_forward(ohlc, ret, cdi, TREINO=tr, TESTE=te)[0]
        comum = r.loc[ret.index[1260]:]       # mesmo período para as oito
        vals.append(sharpe(comum, cdi))
        print(f"   treino {tr}, teste {te}: {vals[-1]:.2f} de {comum.index[0].date()} em diante  (trecho próprio, desde {r.index[0].date()}: {sharpe(r, cdi):.2f})")
print(f"   {len(vals)} combinações: mínimo {min(vals):.2f}, máximo {max(vals):.2f}")
