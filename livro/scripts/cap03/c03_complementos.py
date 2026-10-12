"""Números complementares do Capítulo 3.

Tudo o que o texto cita e não sai de uma listagem impressa nem dos
outros scripts c03_*.py é calculado aqui."""
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(11)
precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()
n, s = len(r), r.std()

print("== moeda: teste de hipótese")
for caras in (60, 55):
    p_exato = stats.binomtest(caras, 100, 0.5).pvalue
    z = (caras - 50) / 5
    print(f"{caras} caras: z = {z:.1f}  p exato = {p_exato:.4f}"
          f"  p normal = {2 * stats.norm.sf(z):.4f}  1/p = {1 / p_exato:.1f}")
# poder contra p = 0,55 com 100 lançamentos (rejeita se <=39 ou >=61)
k = np.arange(101)
rej = np.array([stats.binomtest(int(i), 100, 0.5).pvalue < 0.05 for i in k])
print("região de rejeição:", k[rej][k[rej] < 50].max(), k[rej][k[rej] > 50].min(),
      " tamanho:", stats.binom.pmf(k[rej], 100, 0.5).sum(),
      " poder com p=0,55:", stats.binom.pmf(k[rej], 100, 0.55).sum())

print("== moeda: verossimilhança de 7 caras em 10")
for p in (0.3, 0.5, 0.7, 0.9):
    print(f"p = {p}: p^7 (1-p)^3 = {p ** 7 * (1 - p) ** 3:.5f}")
g = np.linspace(0.01, 0.99, 99)
print("máximo na grade:", g[np.argmax(g ** 7 * (1 - g) ** 3)])

print("== perda em reais de um dia de 5 desvios, R$ 100 mil")
print(f"desvio diário {s:.5f}  5 desvios = R$ {5 * s * 1e5:,.0f}")

print("== nu da t e curtose")
nu = stats.t.fit(r)[0]
print(f"nu = {nu:.3f}  excesso de curtose implícito = {6 / (nu - 4):.2f}")
sub = {"sem março de 2020": r[~((r.index.year == 2020) & (r.index.month == 3))],
       "2010-2019": r[:"2019"], "2021-2025": r["2021":]}
for nome, x in sub.items():
    print(f"{nome:18s} n = {len(x)}  nu = {stats.t.fit(x)[0]:.2f}"
          f"  excesso de curtose = {x.kurt():.2f}")
print("pregões de março de 2020:", n - len(sub["sem março de 2020"]))
bs = [stats.t.fit(rng.choice(r.values, n))[0] for _ in range(300)]
print("nu, bootstrap iid (300): IC 95%", np.percentile(bs, [2.5, 97.5]).round(2))

print("== precisão da volatilidade")
K = stats.kurtosis(r, fisher=False)
vol = s * np.sqrt(252)
print(f"vol anual {vol:.4f}  curtose {K:.2f}")
print(f"erro relativo normal 1/raiz(2n) = {1 / np.sqrt(2 * n):.4f}"
      f" -> {vol / np.sqrt(2 * n) * 100:.2f} p.p.")
print(f"erro relativo com curtose raiz((K-1)/4n) = {np.sqrt((K - 1) / (4 * n)):.4f}"
      f" -> {vol * np.sqrt((K - 1) / (4 * n)) * 100:.2f} p.p.")
x = r.values
for bloco in (21, 63, 126):
    nb = n // bloco + 1
    vs = []
    for _ in range(2000):
        ini = rng.integers(0, n - bloco, nb)
        am = np.concatenate([x[i:i + bloco] for i in ini])[:n]
        vs.append(am.std(ddof=1) * np.sqrt(252))
    print(f"bootstrap em blocos de {bloco}: erro-padrão {np.std(vs) * 100:.2f} p.p."
          f"  IC 95% {np.percentile(vs, [2.5, 97.5]).round(3)}")
ep_media = s / np.sqrt(n) * 252
print(f"erro-padrão da média anual: {ep_media * 100:.2f} p.p.")
va = r.groupby(r.index.year).std() * np.sqrt(252)
print("vol por ano civil: mín", va.min().round(3), va.idxmin(), " máx", va.max().round(3), va.idxmax())

print("== média aritmética (retornos simples)")
R = np.expm1(r)
epR = R.std() / np.sqrt(n)
print(f"média simples anual {252 * R.mean():.4f}  IC 95% "
      f"{252 * (R.mean() - 1.96 * epR):.4f} a {252 * (R.mean() + 1.96 * epR):.4f}"
      f"  t = {R.mean() / epR:.3f}")

print("== poder com o n que dá t esperado de 1,96")
print("poder:", stats.norm.sf(0) + stats.norm.cdf(-2 * 1.96))

print("== variância da autocorrelação amostral com volatilidade agrupada")
rho2 = pd.Series(r.values ** 2).autocorr(1)
d = r - r.mean()
exato = (d.values[1:] ** 2 * d.values[:-1] ** 2).mean() / r.var(ddof=0) ** 2
fator = 1 + (K - 1) * rho2
print(f"K = {K:.2f}  rho_1(r^2) = {rho2:.3f}  1 + (K-1) rho = {fator:.2f}"
      f"  raiz = {np.sqrt(fator):.2f}   E[d_t^2 d_(t-1)^2]/s^4 = {exato:.2f}")
rho1 = r.autocorr(1)
print(f"rho_1 = {rho1:.4f}  t ingênuo = {rho1 * np.sqrt(n):.2f}"
      f"  t corrigido = {rho1 * np.sqrt(n / fator):.2f}")

print("== Bonferroni contra o limite exato sob independência (m = 200)")
m = 200
print(f"alfa/m = {0.05 / m:.6f}   1-(1-alfa)^(1/m) = {1 - 0.95 ** (1 / m):.6f}")

print("== melhor de 200 em 2000 sementes")
mx, nf, nb = [], [], []
tb = stats.norm.isf(0.05 / m / 2)
tc = stats.t.ppf(0.975, 1259)
for sem in range(2000):
    z = np.random.default_rng(sem).normal(0, 0.01, (1260, m))
    t = z.mean(0) / (z.std(0, ddof=1) / np.sqrt(1260))
    mx.append(np.abs(t).max()); nf.append((np.abs(t) > tc).sum())
    nb.append((np.abs(t) > tb).any())
mx, nf = np.array(mx), np.array(nf)
print(f"maior |t|: mediana {np.median(mx):.2f}  faixa de 90% "
      f"{np.percentile(mx, [5, 95]).round(2)}  P(>= 3,44) = {(mx >= 3.44).mean():.3f}")
print(f"razão anual da mediana: {np.median(mx) / np.sqrt(5):.2f}")
print("falsos positivos: faixa de 95%", np.percentile(nf, [2.5, 97.5]),
      " alguma passa de Bonferroni:", np.mean(nb))

print("== correlação na crise: carteira, Fisher, Forbes-Rigobon")
acoes = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "ABEV3.SA",
         "WEGE3.SA", "MGLU3.SA"]
ra = precos[acoes].dropna().pct_change().loc["2015":"2024"]
crise = ra["2020-02-20":"2020-04-30"]
fora = ra.drop(crise.index)
iu = np.triu_indices(7, 1)
cm = lambda x: x.corr().values[iu].mean()
print(f"correlação média: tudo {cm(ra):.3f}  crise {cm(crise):.3f} ({len(crise)} dias)"
      f"  fora {cm(fora):.3f}")
pv = lambda x: x["PETR4.SA"].corr(x["VALE3.SA"])
print(f"PETR4 x VALE3: tudo {pv(ra):.3f}  crise {pv(crise):.3f}  fora {pv(fora):.3f}")
z, ez = np.arctanh(pv(crise)), 1 / np.sqrt(len(crise) - 3)
print("IC de Fisher para a da crise:", np.tanh([z - 1.96 * ez, z + 1.96 * ez]).round(2))
w = np.full(7, 1 / 7)
vc, vf = (crise @ w).var(), (fora @ w).var()
delta = vc / vf - 1
fr = lambda rho: rho / np.sqrt(1 + delta * (1 - rho ** 2))
print(f"variância da carteira igual: crise/fora = {vc / vf:.2f}  delta = {delta:.2f}")
print(f"Forbes-Rigobon: média {cm(crise):.3f} -> {fr(cm(crise)):.3f}"
      f"   PETR4 x VALE3 {pv(crise):.3f} -> {fr(pv(crise)):.3f}")
mi = precos["^BVSP"].dropna().pct_change().loc["2015":"2024"]
print(f"variância do Ibovespa: crise/fora = "
      f"{mi['2020-02-20':'2020-04-30'].var() / mi.drop(mi['2020-02-20':'2020-04-30'].index).var():.2f}")

# carteira de quatro ações do Código 3.7 com as correlações da crise
at = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"]
r4 = precos[at].dropna().pct_change().loc["2015":"2024"]
dp = r4.std().values * np.sqrt(252)
w4 = np.full(4, 0.25)
for nome, c in [("2015-2024", r4.corr().values),
                ("crise", r4["2020-02-20":"2020-04-30"].corr().values)]:
    vcart = np.sqrt(w4 @ (np.outer(dp, dp) * c) @ w4)
    med = c[np.triu_indices(4, 1)].mean()
    print(f"correlações de {nome} (média {med:.2f}): vol. carteira {vcart:.3f}"
          f"  média das vols. {dp.mean():.3f}  redução {1 - vcart / dp.mean():.3f}")
