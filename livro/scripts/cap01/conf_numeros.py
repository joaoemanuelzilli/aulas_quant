"""Refaz os números do Capítulo 1 que não saem dos códigos impressos.

Rode de dentro de livro/scripts/cap01:  python conf_numeros.py
Cada bloco corresponde a um trecho do texto (seção, exemplo ou exercício).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

D = "../../dados/"
aj = pd.read_csv(D + "precos_ajustados.csv", index_col=0, parse_dates=True)
sa = pd.read_csv(D + "precos_sem_ajuste.csv", index_col=0, parse_dates=True)
ATIVOS = ["PETR4.SA", "VALE3.SA", "ABEV3.SA", "WEGE3.SA", "BOVA11.SA"]


def base_anterior(df, ano_base, fim):
    """Série do último fechamento de `ano_base` até o fim de `fim`."""
    s = df.loc[ano_base:fim].dropna()
    return s.loc[s.loc[ano_base].index[-1]:]


print("== Seção de retorno simples: exemplo dos dois pregões")
print(f"20,00 -> 20,50: {20.5 / 20 - 1:.2%}")
print(f"cai 50% e sobe 50%: {0.5 * 1.5:.2f}; para voltar: {1 / 0.5 - 1:.0%}")

print("== Carteira de dois ativos (60% a +10%, 40% a -5%)")
print(f"R$ 1.000: {600 * 1.10 + 400 * 0.95:.0f}; média ponderada {0.6 * 0.10 + 0.4 * -0.05:.1%}")
print(f"em logs: carteira {np.log(1.04):.4f}; média ponderada dos logs "
      f"{0.6 * np.log(1.10) + 0.4 * np.log(0.95):.4f}")

print("== Log-retorno")
print(f"ln 0,5 = {np.log(0.5):.4f}; ln 1,5 = {np.log(1.5):.4f}; soma {np.log(0.75):.4f}; "
      f"exp - 1 = {np.exp(np.log(0.75)) - 1:.0%}; ln 2 = {np.log(2):.3f}")
print(f"20,00 -> 20,50: r = {np.log(20.5 / 20):.4f}")
print(f"R = 1%: R - ln(1+R) = {(0.01 - np.log(1.01)) * 100:.4f} p.p.; R^2/2 = {0.01 ** 2 / 2 * 100:.4f} p.p.")
print(f"R = -29,7%: r = {np.log(1 - 0.297):.3f}")

print("== Ajuste por proventos: exemplo do dividendo")
print(f"retorno pelo preço: {19 / 20 - 1:.0%}; com o dividendo: {(19 + 1) / 20 - 1:.0%}; "
      f"fator {1 - 1 / 20:.2f}; 20 x fator = {20 * 0.95:.2f}")
print(f"JCP de R$ 1,00 a 15%: líquido {1 * 0.85:.2f}; retorno líquido {(19 + 0.85) / 20 - 1:.2%}")
for d in ("2011-01-03",):
    print(f"PETR4 em {d}: sem ajuste {sa.loc[d, 'PETR4.SA']:.2f}; ajustado {aj.loc[d, 'PETR4.SA']:.2f}")
d = "2016-03-14"
print(f"BBAS3 em {d}: sem ajuste {sa.loc[d, 'BBAS3.SA']:.2f}; ajustado {aj.loc[d, 'BBAS3.SA']:.2f}")
a = base_anterior(aj["PETR4.SA"], "2018", "2024")
b = base_anterior(sa["PETR4.SA"], "2018", "2024")
print(f"PETR4 {a.index[0]:%d/%m/%Y} a {a.index[-1]:%d/%m/%Y}: sem ajuste {b.iloc[0]:.2f} -> {b.iloc[-1]:.2f} "
      f"({b.iloc[-1] / b.iloc[0] - 1:.1%}); ajustado {a.iloc[-1] / a.iloc[0] - 1:.1%} (x{a.iloc[-1] / a.iloc[0]:.2f})")
a0 = aj["PETR4.SA"].loc["2019":"2024"].dropna()
b0 = sa["PETR4.SA"].loc["2019":"2024"].dropna()
print(f"  (se a base fosse o primeiro pregão de 2019: {b0.iloc[-1] / b0.iloc[0] - 1:.1%} e {a0.iloc[-1] / a0.iloc[0] - 1:.1%})")

print("== Anualização: exemplo do dia para o ano")
print(f"1,0005^252 - 1 = {1.0005 ** 252 - 1:.1%}; 252 x 0,05% = {252 * 0.0005:.1%}; "
      f"raiz de 252 = {np.sqrt(252):.2f}; 1,5% x raiz = {0.015 * np.sqrt(252):.1%}; 1,5% x 252 = {0.015 * 252:.0%}")

print("== Pregões por ano e convenção de 252")
bv = aj["^BVSP"].dropna()
por_ano = bv.groupby(bv.index.year).size()
print(f"^BVSP 2010-2025: {len(bv)} pregões, {len(bv) / 16:.1f} por ano; mín {por_ano.min()}, máx {por_ano.max()}; "
      f"2022 a 2025: {por_ano.loc[2022:2025].tolist()}")
bcb = pd.read_csv(D + "c02_bcb.csv", index_col=0, parse_dates=True)
cdi = bcb["cdi"].dropna() / 100
print(f"CDI 2010-2025: {len(cdi)} dias úteis, {len(cdi) / 16:.1f} por ano")
print(f"queda de 50% em 10 anos: A=10: {0.5 ** (1 / 10) - 1:.2%}; A=2480/252: {0.5 ** (252 / 2480) - 1:.2%}; "
      f"dobrar: {2 ** (1 / 10) - 1:.2%} e {2 ** (252 / 2480) - 1:.2%}")

print("== Horizonte: mu = 10% e sigma = 20% ao ano, em log, sob normalidade")
mu, sig = 0.10, 0.20
for T in (1, 5, 10, 30):
    dp = sig * np.sqrt(T)
    print(f"T={T:2d}: média {mu * T:.2f}  desvio {dp:.3f}  desvios até zero {mu * T / dp:.2f}  "
          f"P(perda) {norm.cdf(-mu * T / dp):.1%}  R$ 1 vira de {np.exp(mu * T - dp):.2f} a {np.exp(mu * T + dp):.2f} "
          f"(mediana {np.exp(mu * T):.2f})")

print("== Tabela de 2015 a 2024 (mesma amostra do Código l04_anualizar.py)")
p = base_anterior(aj[ATIVOS], "2014", "2024")
log = np.log(p / p.shift(1)).dropna()
anos = (p.index[-1] - p.index[0]).days / 365.25
cresc = p.iloc[-1] / p.iloc[0]
vol = log.std() * np.sqrt(252)
ep = vol / np.sqrt(anos)
r_log = np.log(cresc) / anos
tab = pd.DataFrame({"cagr": cresc ** (1 / anos) - 1, "log a.a.": r_log, "vol": vol, "erro": ep,
                    "ic_inf": np.exp(r_log - 2 * ep) - 1, "ic_sup": np.exp(r_log + 2 * ep) - 1})
print(tab.map(lambda v: f"{v:.1%}"))
pr_todos = aj.loc[p.index[0]:p.index[-1], "^BVSP"].dropna()
print(f"pregões do ^BVSP no período: {len(pr_todos) - 1}; retornos usados: {len(log)}; pregões/ano na amostra: {len(log) / anos:.1f}")
print(f"vol PETR4/WEGE3: {vol['PETR4.SA'] / vol['WEGE3.SA']:.2f}")
for nome, trecho in (("2015-2019", log.loc[:"2019"]), ("2020-2024", log.loc["2020":])):
    v = trecho.std() * np.sqrt(252)
    print(f"  {nome}: PETR4 {v['PETR4.SA']:.1%}  WEGE3 {v['WEGE3.SA']:.1%}  razão {v['PETR4.SA'] / v['WEGE3.SA']:.2f}")
dif = r_log["PETR4.SA"] - r_log["WEGE3.SA"]
ep_dif = (log["PETR4.SA"] - log["WEGE3.SA"]).std() * np.sqrt(252 / anos)
print(f"PETR4 - WEGE3 em log a.a.: {dif:.1%} com erro-padrão {ep_dif:.1%}")
# CDI entre as mesmas datas: a taxa datada do dia d rende de d a d+1
c = cdi.loc[p.index[0]:p.index[-1]].iloc[:-1]
fator = (1 + c).prod()
print(f"CDI de {p.index[0]:%d/%m/%Y} a {p.index[-1]:%d/%m/%Y}: {fator - 1:.1%} acumulado, {fator ** (1 / anos) - 1:.2%} a.a.")
print(f"ABEV3 - CDI: {tab.loc['ABEV3.SA', 'cagr'] - (fator ** (1 / anos) - 1):.1%} a.a.")

print("== Volatilidade anualizada por frequência (armadilha da raiz do tempo e Exercício do desafio)")
linhas = {"diária": vol}
for nome, regra, k in (("semanal", "W-FRI", 52), ("mensal", "ME", 12)):
    q = p.resample(regra).last().dropna()
    linhas[nome] = np.log(q / q.shift(1)).dropna().std() * np.sqrt(k)
freq = pd.DataFrame(linhas)
freq["mensal/diária - 1"] = freq["mensal"] / freq["diária"] - 1
print(freq.map(lambda v: f"{v:.1%}"))
print("autocorrelação de ordem 1 dos log-retornos diários:",
      {a: round(float(log[a].autocorr(1)), 3) for a in ATIVOS})

print(f"erro relativo de uma volatilidade estimada com 120 retornos normais: {1 / np.sqrt(2 * 120):.1%}; "
      f"com {len(log)}: {1 / np.sqrt(2 * len(log)):.1%}")
so = base_anterior(aj["PETR4.SA"], "2014", "2024")
ls = np.log(so / so.shift(1)).dropna()
vs = [ls.std() * np.sqrt(252)]
for regra, k in (("W-FRI", 52), ("ME", 12)):
    q = so.resample(regra).last().dropna()
    vs.append(np.log(q / q.shift(1)).dropna().std() * np.sqrt(k))
print("PETR4 sozinha (sem o dropna dos cinco ativos):", [f"{v:.1%}" for v in vs])

print("== Tabela dos ativos: volatilidade anualizada das sete ações, 2015 a 2024")
acoes = [c for c in aj.columns if c.endswith(".SA") and c != "BOVA11.SA"]
la = np.log(aj[acoes] / aj[acoes].shift(1)).loc["2015":"2024"]
print((la.std() * np.sqrt(252)).sort_values().map(lambda v: f"{v:.1%}").to_dict())
m = aj["MGLU3.SA"].dropna()
print(f"MGLU3: máximo em {m.idxmax():%d/%m/%Y}, {m.max() / m.loc[:'2015'].iloc[-1]:.0f} vezes o fim de 2015; "
      f"fim de 2024: {m.loc[:'2024'].iloc[-1] / m.max() - 1:.0%} do máximo")
print(f"arquivo: {aj.shape[0]} datas, {aj.shape[1]} colunas, de {aj.index[0]:%Y} a {aj.index[-1]:%Y}")

print("== Exercícios")
px = np.array([40.0, 42.0, 39.9, 41.5])
R = px[1:] / px[:-1] - 1
print("1.1 R:", (R * 100).round(2), " r:", np.log(1 + R).round(5), f" acumulado {px[-1] / px[0] - 1:.2%}",
      f"e {np.exp(np.log(1 + R).sum()) - 1:.2%}")
print(f"1.2 perda de 20%: ganho de {0.2 / 0.8:.0%}")
print(f"1.3 dividendo: preço {48.5 / 50 - 1:.1%}; acionista {(48.5 + 2) / 50 - 1:.1%}; fator {1 - 2 / 50:.2f}; "
      f"ajustado {50 * 0.96:.2f} -> {48.5 / 48 - 1:.2%}")
v = base_anterior(aj["VALE3.SA"], "2014", "2024")
lv = np.log(v / v.shift(1)).dropna()
ano_ret = v.resample("YE").last().pct_change().dropna()
ano_vol = lv.groupby(lv.index.year).std() * np.sqrt(252)
print("1.4 VALE3 retorno do ano:", {i.year: f"{x:.1%}" for i, x in ano_ret.items()})
print("    VALE3 vol do ano:", {i: f"{x:.1%}" for i, x in ano_vol.items()}, "-> maior em", ano_vol.idxmax())
print(f"1.5 +50% e -50% em partes iguais: carteira {np.log(1 + 0.5 * 0.5 - 0.5 * 0.5):.4f}; "
      f"média dos logs {0.5 * (np.log(1.5) + np.log(0.5)):.4f}")
ps = base_anterior(sa[ATIVOS], "2014", "2024").loc[p.index]
cs = ps.iloc[-1] / ps.iloc[0]
print("1.6 CAGR sem ajuste:", {a: f"{x:.1%}" for a, x in (cs ** (1 / anos) - 1).items()})
Rp = np.exp(log["PETR4.SA"]) - 1
G = np.exp(log["PETR4.SA"].mean()) - 1
print(f"1.7 PETR4: média {Rp.mean():.4%}  desvio {Rp.std():.3%}  G {G:.4%}  média - var/2 {Rp.mean() - Rp.var() / 2:.4%}")
print("1.8 PETR4 diária/semanal/mensal:", [f"{freq.loc['PETR4.SA', k]:.1%}" for k in ("diária", "semanal", "mensal")])
