"""Testes de robustez da regra de Donchian com walk-forward.

Não é impresso no livro; gera os números da seção de robustez
e grava dados/c08_robustez_ativos.csv e dados/c08_robustez_n.csv
para a figura.
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


def walk_forward(ohlc, ret, cdi, custo=0.001, defasagem=1, grade=GRADE, TREINO=TREINO, TESTE=TESTE, desloc=0):
    """Mesmo procedimento do l07 (janela rolante), para qualquer ativo."""
    alvos = pd.DataFrame({n: donchian(ohlc, n) for n in grade})
    rets = alvos.apply(lambda a: backtest(a, ret, cdi, custo, defasagem)[0])
    costura = pd.Series(0.0, index=ret.index)
    for ini in range(TREINO + desloc, len(ret), TESTE):
        treino = rets.iloc[ini - TREINO:ini]
        n = treino.apply(sharpe, cdi=cdi).idxmax()
        fim = min(ini + TESTE, len(ret))
        costura.iloc[ini - 1:fim - 1] = alvos[n].iloc[ini - 1:fim - 1]
    r, giro = backtest(costura, ret, cdi, custo, defasagem)
    return r.iloc[TREINO + desloc:], rets, costura


ohlc, ret, cdi = carregar_ohlc()

# 1. parâmetro: Sharpe de cada n fixo, amostra inteira e período fora da amostra
wf, rets, _ = walk_forward(ohlc, ret, cdi)
sr_tudo = rets.apply(sharpe, cdi=cdi)
sr_fora = rets.iloc[TREINO:].apply(sharpe, cdi=cdi)
print("1. Sharpe por n (amostra inteira):", sr_tudo.round(2).to_dict())
print("   Sharpe por n (2013-2025):     ", sr_fora.round(2).to_dict())
print(f"   n com Sharpe > 0,30 na amostra inteira: {[n for n in GRADE if sr_tudo[n] > 0.30]}")
print(f"   walk-forward: {sharpe(wf, cdi):.2f}")

# 2. custos e defasagem (1 = execução no fechamento do sinal; 2 = um pregão de atraso)
print("2. Sharpe do walk-forward por custo e defasagem")
linhas = []
for custo in (0, 0.0005, 0.001, 0.002, 0.005):
    s1 = sharpe(walk_forward(ohlc, ret, cdi, custo, 1)[0], cdi)
    s2 = sharpe(walk_forward(ohlc, ret, cdi, custo, 2)[0], cdi)
    linhas.append((custo, s1, s2))
    print(f"   custo {custo:.2%}: defasagem 1 -> {s1:.2f}   defasagem 2 -> {s2:.2f}")

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
fora = df[df.ativo != "PETR4"]
print(f"   sem PETR4 (ativo de desenvolvimento): supera em {(fora.wf > fora.comprado).sum()} de {len(fora)}; médias WF {fora.wf.mean():.2f}, comprado {fora.comprado.mean():.2f}")
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

# 6. deslocamento do início das janelas (treino 756, teste 252), medido no trecho comum
print("6. Sharpe do walk-forward deslocando o início das janelas (pregões)")
comum0 = ret.index[TREINO + 252]
vd = []
for d in range(0, 253, 21):
    r = walk_forward(ohlc, ret, cdi, desloc=d)[0]
    vd.append(sharpe(r.loc[comum0:], cdi))
    print(f"   deslocamento {d:3d}: {vd[-1]:.2f} de {comum0.date()} em diante  (trecho próprio, desde {r.index[0].date()}: {sharpe(r, cdi):.2f})")
bh = sharpe(ret.loc[comum0:], cdi)
print(f"   {len(vd)} deslocamentos: mínimo {min(vd):.2f}, máximo {max(vd):.2f}, mediana {np.median(vd):.2f}; abaixo de PETR4 comprada ({bh:.2f}) em {sum(v < bh for v in vd)}")

# 7. custo x execução com o caminho de parâmetros FIXO (o da configuração-base)
print("7. Sharpe com a costura da configuração-base, mudando só custo e defasagem")
_, _, costura_base = walk_forward(ohlc, ret, cdi)
for custo in (0, 0.0005, 0.001, 0.002, 0.005):
    s1 = sharpe(backtest(costura_base, ret, cdi, custo, 1)[0].iloc[TREINO:], cdi)
    s2 = sharpe(backtest(costura_base, ret, cdi, custo, 2)[0].iloc[TREINO:], cdi)
    print(f"   custo {custo:.2%}: defasagem 1 -> {s1:.2f}   defasagem 2 -> {s2:.2f}   diferença {s1 - s2:.2f}")
for d in (1, 2):    # o mesmo efeito em retorno anual, no custo-padrão
    r = backtest(costura_base, ret, cdi, 0.001, d)[0].iloc[TREINO:]
    print(f"   custo 0.10%, defasagem {d}: retorno {(1 + r).prod() ** (252 / len(r)) - 1:.1%} a.a.")

# 8. incerteza: bootstrap em blocos de 21 pregões do Sharpe e da diferença para PETR4 comprada
print("8. bootstrap em blocos (21 pregões, 5000 reamostras, semente 8)")
rng = np.random.default_rng(8)
ex_wf = (wf - cdi.loc[wf.index]).values
ex_bh = (ret.loc[wf.index] - cdi.loc[wf.index]).values
T, B = len(ex_wf), 21
sr = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)
a_, d_ = [], []
for _ in range(5000):
    ini = rng.integers(0, T - B + 1, size=T // B + 1)
    idx = (ini[:, None] + np.arange(B)).ravel()[:T]
    a_.append(sr(ex_wf[idx])); d_.append(sr(ex_wf[idx]) - sr(ex_bh[idx]))
a_, d_ = np.array(a_), np.array(d_)
print(f"   Sharpe do WF: {sr(ex_wf):.2f}; desvio-padrão {a_.std():.2f}; IC 95% {np.percentile(a_, 2.5):.2f} a {np.percentile(a_, 97.5):.2f}")
print(f"   diferença para PETR4 comprada: {sr(ex_wf) - sr(ex_bh):.2f}; desvio-padrão {d_.std():.2f}; IC 95% {np.percentile(d_, 2.5):.2f} a {np.percentile(d_, 97.5):.2f}; negativa em {(d_ < 0).mean():.0%}")

# 9. dispersão dos Sharpes das janelas de teste (cada uma tem cerca de um ano)
from l07_walkforward import walk_forward as wf_l07
tab = wf_l07()[1]
print(f"9. SR teste das 13 janelas: média {tab['SR teste'].mean():.2f}, desvio-padrão {tab['SR teste'].std():.2f}, "
      f"mínimo {tab['SR teste'].min():.2f}, máximo {tab['SR teste'].max():.2f}; negativos: {(tab['SR teste'] < 0).sum()}")
