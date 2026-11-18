"""Refaz os números do Capítulo 3 que aparecem no texto, nas legendas e nos
gabaritos, e termina com erro se algum deixar de valer.

Rode de dentro de livro/scripts/cap03:  python conf_numeros.py
"""
from math import comb, factorial

import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import binom, norm, poisson

D = "../../dados/"
aj = pd.read_csv(D + "precos_ajustados.csv", index_col=0, parse_dates=True)
falhas = []


def confere(nome, valor, texto, casas):
    """`texto` é o número como impresso no livro; compara arredondando."""
    ok = round(float(valor), casas) == round(texto, casas)
    print(f"{'ok ' if ok else 'ERRO'} {nome}: {valor:.6g} (texto {texto})")
    if not ok:
        falhas.append(nome)


print("== 3.1 Dois dados")
confere("P(soma 7)", 6 / 36, 1 / 6, 6)
confere("P(pelo menos um 6)", 1 - 25 / 36, 0.306, 3)
confere("união 1/6+1/6-1/36", 1 / 6 + 1 / 6 - 1 / 36, 11 / 36, 9)

print("== 3.2 Condicional")
confere("P(6 | par)", (1 / 6) / (1 / 2), 1 / 3, 9)
confere("urna sem reposição", 3 / 5 * 2 / 4, 0.3, 9)
p = aj[["PETR4.SA", "VALE3.SA"]].dropna()
R = p.pct_change().dropna().loc["2015":"2024"]
A, B = R["PETR4.SA"] > 0, R["VALE3.SA"] > 0
confere("pregões", len(R), 2479, 0)
confere("P(A)", A.mean(), 0.523, 3)
confere("P(B)", B.mean(), 0.498, 3)
confere("P(A e B)", (A & B).mean(), 0.323, 3)
confere("P(A)P(B)", A.mean() * B.mean(), 0.260, 3)
confere("P(A|B)", A[B].mean(), 0.649, 3)
confere("P(A|não B)", A[~B].mean(), 0.398, 3)
confere("P(A ou B)", (A | B).mean(), 0.698, 3)
confere("união pela fórmula com valores impressos", 0.523 + 0.498 - 0.323, 0.698, 3)
confere("P(A|B) - P(A|não B), pontos", 100 * (A[B].mean() - A[~B].mean()), 25, 0)
confere("prob. total com valores impressos", 0.649 * 0.498 + 0.398 * 0.502, 0.523, 3)
Rp = aj["PETR4.SA"].dropna().pct_change().dropna().loc["2015":"2024"]
up = Rp > 0
ontem = up.shift(1)
m = ontem.notna()
apos_alta, apos_nao = up[m & (ontem == True)], up[m & (ontem == False)]
confere("dias após alta", len(apos_alta), 1296, 0)
confere("P(alta | alta ontem)", 100 * apos_alta.mean(), 50.5, 1)
confere("dias após não alta", len(apos_nao), 1182, 0)
confere("P(alta | não alta ontem)", 100 * apos_nao.mean(), 54.3, 1)

print("== 3.3 Bayes")
ib = aj["^BVSP"].dropna()
r = np.log(ib / ib.shift(1)).dropna()
base = (r < -0.03).mean()
confere("dias de queda", (r < -0.03).sum(), 76, 0)
confere("pregões do Ibovespa", len(r), 3963, 0)
confere("taxa-base %", 100 * base, 1.92, 2)


def post(pr, s, f):
    return s * pr / (s * pr + f * (1 - pr))


confere("P(queda | alarme) %", 100 * post(base, 0.8, 0.1), 13.5, 1)
folhas = [base * 0.8, base * 0.2, (1 - base) * 0.1, (1 - base) * 0.9]
for v, t in zip(folhas, (0.0153, 0.0038, 0.0981, 0.8827)):
    confere("folha da árvore", v, t, 4)
confere("ramo 0,9808", 1 - base, 0.9808, 4)
confere("P(alarme)", folhas[0] + folhas[2], 0.1134, 4)
confere("0,0153/0,1134 %", 100 * 0.0153 / 0.1134, 13.5, 1)
confere("mil pregões: quedas", 1000 * base, 19, 0)
confere("mil pregões: quedas com alarme", 800 * base, 15, 0)
confere("mil pregões: normais", 1000 * (1 - base), 981, 0)
confere("mil pregões: alarmes falsos", 100 * (1 - base), 98, 0)
confere("15 de 113 %", 100 * 15 / 113, 13.3, 1)
confere("alarme falso 2% %", 100 * post(base, 0.8, 0.02), 43.9, 1)
confere("P(real|passou) pi=10% %", 100 * post(0.10, 0.5, 0.05), 52.6, 1)
confere("P(real|passou) pi=1% %", 100 * post(0.01, 0.5, 0.05), 9.2, 1)

print("== 3.4 Esperança")
x = np.arange(1, 7)
confere("E[dado]", x.mean(), 3.5, 4)
u = sp.symbols("u")
confere("E[uniforme]", float(sp.integrate(u, (u, 0, 1))), 0.5, 6)
confere("Var[uniforme]", float(sp.integrate(u**2, (u, 0, 1)) - sp.Rational(1, 4)), 1 / 12, 6)
W2 = sorted([1.5 * 1.5, 1.5 * 0.5, 0.5 * 1.5, 0.5 * 0.5])
confere("E[W2]", np.mean(W2), 1.0, 6)
confere("mediana W2", np.median(W2), 0.75, 6)
elw = 0.5 * (np.log(1.5) + np.log(0.5))
confere("E[ln W1]", elw, -0.144, 3)
confere("1/2 ln 0,75", 0.5 * np.log(0.75), -0.144, 3)

print("== 3.5 Variância")
confere("Var dado", (x**2).mean() - x.mean() ** 2, 2.9167, 4)
confere("35/12", 35 / 12, 2.9167, 4)
confere("ativo +21%/-19% média %", 100 * (0.21 - 0.19) / 2, 1, 6)
cov_ind = (A & B).mean() - A.mean() * B.mean()
confere("Cov indicadoras", cov_ind, 0.063, 3)
confere("Corr indicadoras", np.corrcoef(A, B)[0, 1], 0.25, 2)
X, Y = R["PETR4.SA"], R["VALE3.SA"]
k = np.sqrt(252)
confere("média das vols %", 100 * (X.std() + Y.std()) / 2 * k, 45.5, 1)
confere("vol carteira %", 100 * (0.5 * X + 0.5 * Y).std() * k, 38.7, 1)

print("== 3.6 Binomial e Poisson")
confere("C(5,3)/32 %", 100 * comb(5, 3) / 32, 31.25, 2)
pet = aj["PETR4.SA"].dropna()
alta = (pet.pct_change().dropna() > 0).to_numpy()
confere("pregões PETR4 2010-2025", len(alta), 3966, 0)
confere("pregões fora dos blocos", len(alta) - 188 * 21, 18, 0)
confere("P(K>=7) aposta %", 100 * (120 + 45 + 10 + 1) / 1024, 17.2, 1)
confere("P(K>=7) binom %", 100 * binom.sf(6, 10, 0.5), 17.2, 1)
confere("perde %", 100 * binom.cdf(6, 10, 0.5), 83, 0)
confere("0,75^5", 0.75**5, 0.237, 3)
confere("W com 5 caras", 1.5**5 * 0.5**5, 0.237, 3)
confere("VaR: média", 250 * 0.01, 2.5, 6)
confere("VaR: P(0) binomial %", 100 * binom.pmf(0, 250, 0.01), 8.11, 2)
confere("VaR: P(0) Poisson %", 100 * np.exp(-2.5), 8.21, 2)
N = (r < -0.03).groupby(r.index.to_period("M")).sum()
lam = N.mean()
confere("variância/média", N.var() / lam, 2.4, 1)
confere("P(N>=9) x 1e10", poisson.sf(8, lam) * 1e10, 4.6, 1)
confere("um mês a cada (milhões de anos)", 1 / poisson.sf(8, lam) / 12 / 1e6, 180, -1)
confere("Poisson espera >=4 em 192 meses", len(N) * poisson.sf(3, lam), 0.1, 1)
normal_prev = len(r) * norm.cdf(-0.03, r.mean(), r.std())
confere("normal prevê dias abaixo de -3%", normal_prev, 77, 0)
e = (r < -0.03).to_numpy()
idx = np.where(e)[0]
seg = np.array([e[i + 1:i + 6].any() for i in idx])
confere("seguidos em 5 pregões %", 100 * seg.mean(), 30, 0)
taxa = e.mean()
confere("Poisson: seguidos em 5 (por pregão) %", 100 * (1 - np.exp(-5 * taxa)), 9, 0)
confere("intervalo mediano entre quedas", np.median(np.diff(idx)), 13, 0)
confere("mediana da espera de Poisson", np.log(2) / taxa, 36, 0)

print("== 3.7 Normal")
for kk, t in ((1, 68.3), (2, 95.4), (3, 99.7)):
    confere(f"P(|Z|<{kk}) %", 100 * (2 * norm.cdf(kk) - 1), t, 1)
confere("ln 0,70", np.log(0.7), -0.357, 3)
confere("z", (np.log(0.7) - 0.1) / 0.48, -0.951, 3)
confere("P(queda > 30%) %", 100 * norm.cdf((np.log(0.7) - 0.1) / 0.48), 17.1, 1)
confere("quantil 5%", norm.ppf(0.05), -1.645, 3)
confere("3% x raiz(252) %", 3 * np.sqrt(252), 47.6, 1)
pb = pet.loc["2014":"2024"]
pb = pb.loc[pb.loc["2014"].index[-1]:]
lr = np.log(pb / pb.shift(1)).dropna()
anos = (pb.index[-1] - pb.index[0]).days / 365.25
vol = lr.std() * np.sqrt(252)
confere("vol PETR4 2015-2024 (Cap. 1) %", 100 * vol, 48, 0)
confere("erro-padrão da média (Cap. 1), pontos", 100 * vol / np.sqrt(anos), 15, 0)

print("== 3.8 Lognormal")
mu, s, P0 = 0.10, 0.48, 30.0
confere("mediana", P0 * np.exp(mu), 33.16, 2)
confere("média", P0 * np.exp(mu + s**2 / 2), 37.20, 2)
confere("P(P > média) %", 100 * norm.sf(s / 2), 40.5, 1)
confere("mediana 5 anos", P0 * np.exp(5 * mu), 49.46, 2)
confere("média 5 anos", P0 * np.exp(5 * mu + 5 * s**2 / 2), 87.99, 2)
xs, ms, ss = sp.symbols("x m s", real=True)
dif = sp.expand(xs - (xs - ms) ** 2 / (2 * ss**2) - (-(xs - ms - ss**2) ** 2 / (2 * ss**2) + ms + ss**2 / 2))
assert dif == 0, "completar o quadrado"
from scipy.integrate import quad
for m0, s0 in ((0.10, 0.48), (0.5, 1.0733), (-0.2, 0.3)):
    Ee = quad(lambda t: np.exp(t) * norm.pdf(t, m0, s0), m0 - 12 * s0, m0 + 12 * s0)[0]
    assert np.isclose(Ee, np.exp(m0 + s0**2 / 2), rtol=1e-9), "E[e^r]"
print("ok  completamento de quadrado (sympy) e E[e^r] = e^(m+s^2/2) (integração numérica)")

print("== 3.9 LGN e TCL")
freq = np.cumsum(alta) / np.arange(1, len(alta) + 1)
for n, t in ((20, 35), (100, 50), (1000, 49.8), (len(alta), 51.3)):
    confere(f"frequência após {n}", 100 * freq[n - 1], t, 0 if n < 1000 else 1)
pa = alta.mean()
confere("dois erros-padrão, pontos", 200 * np.sqrt(pa * (1 - pa) / len(alta)), 1.6, 1)
um = np.full(6, 1 / 6)
pmf = np.array([1.0])
for n in range(1, 6):
    pmf = np.convolve(pmf, um)
    vals = np.arange(n, 6 * n + 1)
    assert np.isclose((vals * pmf).sum(), 3.5 * n)
    assert np.isclose(((vals - 3.5 * n) ** 2 * pmf).sum(), 35 * n / 12)
print("ok  média 3,5n e variância 35n/12 da soma de dados")

print("== 3.10 Monte Carlo")
ex = norm.cdf((np.log(0.7) - mu) / s)
ep = np.sqrt(ex * (1 - ex) / 10_000)
for ep_ in (ep, np.sqrt(0.1778 * (1 - 0.1778) / 10_000)):   # EP exato e com p chapéu
    desvio = (0.1778 - ex) / ep_
    confere("desvio em erros-padrão (N = 10 mil)", desvio, 1.9, 1)
    assert 0.055 < 2 * norm.sf(desvio) < 0.065, "cerca de 6% das rodadas"
confere("erro com 1 milhão", np.sqrt(ex * (1 - ex) / 1e6), 0.0004, 4)

print("== Exercícios")
confere("3.1 P(7 | pelo menos um 6)", (2 / 36) / (11 / 36), 2 / 11, 9)
print("    independentes?", np.isclose(2 / 36, (6 / 36) * (11 / 36)), "(não)")
confere("3.2 com reposição", (3 / 5) ** 2, 0.36, 9)
confere("3.2 segunda vermelha sem reposição", 3 / 5 * 2 / 4 + 2 / 5 * 3 / 4, 0.6, 9)
aniv = lambda n: 1 - np.prod((365 - np.arange(n)) / 365)
confere("3.3 23 pessoas", aniv(23), 0.507, 3)
confere("3.3 menor n acima de 50%", min(n for n in range(1, 60) if aniv(n) > 0.5), 23, 0)
confere("3.4 duas caras", 1 / (1 + 1 / 8), 8 / 9, 9)
confere("3.4 quarto lançamento", 8 / 9 + 1 / 9 * 0.5, 17 / 18, 9)
confere("3.5 E[N]", sum(kk / 2**kk for kk in range(1, 200)), 2, 9)
rng = np.random.default_rng(seed=3)
geo = rng.geometric(0.5, 100_000)
print(f"    simulação: {geo.mean():.4f} ± {geo.std() / np.sqrt(geo.size):.4f}")
confere("3.6 P(0) binomial %", 100 * binom.pmf(0, 250, 0.01), 8.1, 1)
confere("3.6 P(>=7) binomial %", 100 * binom.sf(6, 250, 0.01), 1.37, 2)
confere("3.6 P(0) Poisson %", 100 * poisson.pmf(0, 2.5), 8.2, 1)
confere("3.6 P(>=7) Poisson %", 100 * poisson.sf(6, 2.5), 1.42, 2)
z = aj[["ITUB4.SA", "BBAS3.SA"]].dropna().pct_change().dropna().loc["2015":"2024"]
a, b = z["ITUB4.SA"], z["BBAS3.SA"]
vls = np.sqrt(a.var() + b.var() - 2 * a.cov(b)) * k
confere("3.7 vol ITUB4 %", 100 * a.std() * k, 30.5, 1)
confere("3.7 vol BBAS3 %", 100 * b.std() * k, 40.5, 1)
confere("3.7 correlação", a.corr(b), 0.759, 3)
confere("3.7 long-short pela fórmula %", 100 * vls, 26.4, 1)
confere("3.7 long-short pela série %", 100 * (a - b).std() * k, 26.4, 1)
confere("3.8 dobrar em 1 ano %", 100 * norm.sf((np.log(2) - mu) / s), 10.8, 1)
confere("3.8 dobrar em 5 anos %", 100 * norm.sf((np.log(2) - 5 * mu) / (s * np.sqrt(5))), 42.9, 1)
rng = np.random.default_rng(seed=1)
cam = rng.normal(mu / 252, s / np.sqrt(252), size=(100_000, 252)).cumsum(axis=1)
toque = (cam.min(axis=1) <= np.log(0.7)).mean()
eb = np.sqrt(toque * (1 - toque) / cam.shape[0])
bb = np.log(0.7)
cont = norm.cdf((bb - mu) / s) + np.exp(2 * mu * bb / s**2) * norm.cdf((bb + mu) / s)
print(f"3.9 barreira diária: {toque:.4f} ± {eb:.4f}; terminal "
      f"{(cam[:, -1] < bb).mean():.4f}; monitoramento contínuo (fórmula) {cont:.4f}")
confere("3.9 barreira %", 100 * toque, 36.5, 1)
confere("3.9 contínua %", 100 * cont, 38.8, 1)

print("== revisão v10")
confere("alarme com sensibilidade 100% %", 100 * post(base, 1.0, 0.1), 16.4, 1)
confere("crescimento contínuo do preço esperado %", 100 * (0.10 + 0.48**2 / 2), 21.5, 1)
confere("ln 0,75^50", 50 * np.log(0.75), -14.4, 1)
assert 0.75**50 < 1e-6

assert not falhas, f"números que não conferem: {falhas}"
print("todos os números conferem")
