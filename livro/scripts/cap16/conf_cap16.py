"""Conferência dos números do Capítulo 16.

Refaz cada conta citada no texto, nas tabelas, nas legendas e nos
gabaritos, e confere que as saídas coladas no capítulo são idênticas
à execução dos scripts. Termina com erro se alguma deixar de valer.
Rode de dentro de scripts/cap16/ (leva cerca de um minuto).
"""
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import calculo_algebra as ca
import combinatoria as cb
import probabilidade as pr
from algoritmos import (dois_somam, drawdown_maximo, media_movel,
                        media_movel_ingenua, media_movel_np,
                        melhor_negocio)

AQUI = Path(__file__).resolve().parent
TEX = (AQUI.parents[1] / "capitulos" / "cap16.tex").read_text("utf8")
falhas = []


def confere(nome, cond):
    print(("ok    " if cond else "FALHA ") + nome)
    if not cond:
        falhas.append(nome)


def perto(a, b, tol):
    return abs(a - b) <= tol


# ---- saídas coladas == execução ---------------------------------
blocos = re.findall(r"\\begin\{lstlisting\}\[style=saida\]\n(.*?)"
                    r"\\end\{lstlisting\}", TEX, re.S)
for script in ("l01_bova11.py", "l02_bayes.py"):
    saida = subprocess.run([sys.executable, script], cwd=AQUI,
                           capture_output=True, text=True,
                           check=True).stdout
    confere(f"saída de {script} colada no texto", saida in blocos)

# ---- combinatória -----------------------------------------------
confere("caminhos 210 de 1.024",
        cb.caminhos(10, 6) == cb.caminhos_forca_bruta(10, 6) == 210)
confere("20,5%", round(210 / 1024 * 100, 1) == 20.5)
confere("grade 5x5 = 252", cb.caminhos_grade(5, 5) == 252
        == cb.caminhos(10, 5))
confere("Pascal", math.comb(10, 6) == math.comb(9, 5) + math.comb(9, 6))
confere("estrelas e barras 286 e 84",
        cb.estrelas_barras(10, 4) == 286
        == cb.estrelas_barras_forca_bruta(10, 4)
        and cb.estrelas_barras(10, 4, 1) == 84
        == cb.estrelas_barras_forca_bruta(10, 4, 1))
confere("4^10 = 1.048.576", 4 ** 10 == 1_048_576)
confere("um par: 1.098.240 de 2.598.960 = 42,26%",
        cb.um_par() == 1_098_240 == cb.um_par_forca_bruta()
        and math.comb(52, 5) == 2_598_960
        and round(cb.um_par() / math.comb(52, 5) * 100, 2) == 42.26)
confere("48*44*40/6 = 14.080 = 220*64",
        48 * 44 * 40 // 6 == 14_080 == 220 * 64
        and math.comb(12, 3) == 220)
confere("D_8 = 14.833 (40.320 permutações)",
        cb.desarranjos(8) == 14_833 == cb.desarranjos_forca_bruta(8)
        and math.factorial(8) == 40_320)
p10 = cb.desarranjos(10) / math.factorial(10)
confere("D_10 = 1.334.961, 0,367879 = 1/e",
        cb.desarranjos(10) == 1_334_961 and round(p10, 6) == 0.367879
        and round(1 / math.e, 6) == 0.367879)
confere("desarranjo simulado 0,3670",
        round(cb.desarranjo_simulado(10), 4) == 0.3670)

# ---- probabilidade ----------------------------------------------
confere("dado 4,25 e 14/3", pr.dado_segunda_chance() == 4.25
        and perto(pr.dado_tres_jogadas(), 14 / 3, 1e-12))
m, ep = pr.dado_simulado()
confere("dado simulado 4,2488 (0,0015)",
        round(m, 4) == 4.2488 and round(ep, 4) == 0.0015)
m16, ep16 = pr.dado_simulado(16)
zs = np.abs(pr.dado_sementes())
confere("semente 16: 4,2548, 3,1 ep; 400 sementes: 4,5% > 2 ep, "
        "só a 16 > 3 ep",
        round(m16, 4) == 4.2548 and round((m16 - 4.25) / ep16, 1) == 3.1
        and round(np.mean(zs > 2) * 100, 1) == 4.5
        and list(np.flatnonzero(zs > 3)) == [16]
        and round(2 * stats.norm.sf(2) * 100) == 5
        and round(2 * stats.norm.sf(3) * 100, 1) == 0.3)
mcc, ecc = pr.espera_simulada("CC")
mck, eck = pr.espera_simulada("CK")
confere("CC 5,988 (0,010) e CK 4,008 (0,004)",
        (round(mcc, 3), round(ecc, 3), round(mck, 3), round(eck, 3))
        == (5.988, 0.010, 4.008, 0.004))
pv = pr.bayes_moeda()
confere("Bayes 0,5062 = 1024/2023 e 0,7531",
        perto(pv, 1024 / 2023, 1e-12) and round(pv, 4) == 0.5062
        and round(pv + 0.5 * (1 - pv), 4) == 0.7531)
ps, eps, n10 = pr.bayes_moeda_simulada()
confere("Bayes simulado 0,5144 (0,0080), 3.892 casos, ~1 ep",
        round(ps, 4) == 0.5144 and round(eps, 4) == 0.0080
        and n10 == 3892 and abs(ps - pv) < 1.1 * eps)
confere("ruína a=4, b=2: 1/3 e 8 passos; lucro 0",
        pr.ruina(4, 2) == (1 / 3, 8)
        and perto(4 / 3 - 2 * 2 / 3, 0, 1e-12))
confere("ruína simulada 0,3323 e 8,02",
        tuple(round(x, 2 + 2 * (i == 0)) for i, x in
              enumerate(pr.ruina_simulada(4, 2))) == (0.3323, 8.02))
confere("ruína simulada a=b=10: 0,4979 e 100,16",
        tuple(round(x, 4 - 2 * i) for i, x in
              enumerate(pr.ruina_simulada(10, 10))) == (0.4979, 100.16))
confere("m_k = (a-k)(k+b) resolve a recorrência",
        all((4 - k) * (k + 2) == 1 + 0.5 * (4 - k - 1) * (k + 1 + 2)
            + 0.5 * (4 - k + 1) * (k - 1 + 2) for k in range(-1, 4)))
confere("máximo de uniformes: 10/11 = 0,9091 e 100/101 = 0,9901",
        round(pr.maximo_uniformes(10), 4) == 0.9091
        and round(pr.maximo_uniformes(100), 4) == 0.9901)
m10, e10 = pr.maximo_uniformes_simulado(10)
m100, _ = pr.maximo_uniformes_simulado(100)
confere("máximo simulado 0,9090 (0,0002) e 0,9901",
        round(m10, 4) == 0.9090 and round(e10, 4) == 0.0002
        and round(m100, 4) == 0.9901)

# ---- cálculo e álgebra linear -----------------------------------
from scipy import integrate, stats  # noqa: E402
mp = ca.media_parte_positiva()
num, _ = integrate.quad(lambda z: z * stats.norm.pdf(z), 0, np.inf)
confere("E[max(Z,0)] = 1/raiz(2 pi) = 0,3989 (sympy e quad)",
        perto(float(mp), 1 / math.sqrt(2 * math.pi), 1e-15)
        and perto(num, float(mp), 1e-10) and round(num, 4) == 0.3989)
confere("call: 11,97 aprox, 11,92 BS; 3 meses 5,98 e 5,98",
        round(ca.call_aproximada(100, 0.3, 1), 2) == 11.97
        and round(ca.call_bs_sem_juros(100, 0.3, 1), 2) == 11.92
        and round(ca.call_aproximada(100, 0.3, 0.25), 2) == 5.98
        and round(ca.call_bs_sem_juros(100, 0.3, 0.25), 2) == 5.98)
d1 = 0.3 / 2
confere("BS com r=0, K=S coincide com a fórmula geral",
        perto(ca.call_bs_sem_juros(100, 0.3, 1),
              100 * stats.norm.cdf(d1) - 100 * stats.norm.cdf(d1 - 0.3),
              1e-12))
confere("série de Phi: termo cúbico -x^3/6",
        perto(stats.norm.cdf(0.1) - 0.5,
              (0.1 - 0.1 ** 3 / 6) / math.sqrt(2 * math.pi), 1e-6))
confere("regra dos 72: 9,006 / 9,011 / 8,66; 6,116 / 6,123; 72,1",
        round(ca.dobra_exato(0.08), 3) == 9.006
        and round(ca.dobra_taylor(0.08), 3) == 9.011
        and round(0.693 / 0.08, 2) == 8.66
        and round(ca.dobra_exato(0.12), 3) == 6.116
        and round(ca.dobra_taylor(0.12), 3) == 6.123
        and round(69.3 * 1.04, 1) == 72.1
        and round(math.log(2), 3) == 0.693
        and round(math.log(2) / 2, 2) == 0.35)
confere("contínua: ln2/r exato", perto(
    math.log(2) / 0.08, math.log(2) / math.log(math.exp(0.08)), 1e-12))
lo, hi = ca.faixa_correlacao(0.9, 0.9)
nlo, nhi = ca.faixa_correlacao_numerica(0.9, 0.9)
confere("faixa [0,62; 1] por fórmula, autovalores e cos 2theta",
        perto(lo, 0.62, 1e-12) and perto(hi, 1, 1e-12)
        and perto(nlo, 0.62, 1e-4) and perto(nhi, 1, 1e-4)
        and perto(math.cos(2 * math.acos(0.9)), 0.62, 1e-12))
lo5, hi5 = ca.faixa_correlacao(0.5, 0.5)
confere("exercício: faixa [-0,5; 1]",
        perto(lo5, -0.5, 1e-12) and perto(hi5, 1, 1e-12))
ev = ca.equicorrelacao(5, 0.3)
confere("autovalores 2,2 e 4 x 0,7",
        np.allclose(np.sort(ev), [0.7] * 4 + [2.2]))
confere("menor rho = -1/(n-1)",
        ca.equicorrelacao(5, -0.25).min() > -1e-12
        and ca.equicorrelacao(5, -0.26).min() < 0)
v1, v2 = ca.variancia_pesos_iguais(5, 0.3)
confere("variância de pesos iguais", perto(v1, v2, 1e-12))

# ---- algoritmos e dados de BOVA11 -------------------------------
px = pd.read_csv(AQUI.parents[1] / "dados" / "precos_ajustados.csv",
                 index_col=0, parse_dates=True)["BOVA11.SA"].dropna()
confere("3.967 pregões e 7.866.561 pares",
        len(px) == 3967 and 3967 * 3966 // 2 == 7_866_561)
lucro, i, j = melhor_negocio(list(np.log(px)))
dd = px / px.cummax() - 1
vale = dd.idxmin()
pico = px[:vale].idxmax()
confere("vale do drawdown = compra do melhor negócio (26/01/2016)",
        vale == px.index[i] == pd.Timestamp("2016-01-26")
        and pico == pd.Timestamp("2010-11-04")
        and round(px[pico], 2) == 72.40)
confere("drawdown = pior negócio",
        perto(drawdown_maximo(list(px)),
              -(1 - np.exp(-melhor_negocio(list(-np.log(px)))[0])),
              1e-12))
confere("dois_somam bordas", dois_somam([3], 6) is None
        and dois_somam([3, 3], 6) == (0, 1))
v = list(px)
mm = np.array(media_movel_ingenua(v, 252))
dif = max(np.abs(np.array(media_movel(v, 252)) - mm).max(),
          np.abs(media_movel_np(np.array(v), 252) - mm).max(),
          np.abs(px.rolling(252).mean().dropna().values - mm).max())
confere("médias móveis diferem por menos de 2e-12", dif < 2e-12)
confere("20 cortes para 10^6", math.ceil(math.log2(10 ** 6 + 1)) == 20
        and 2 ** 20 > 10 ** 6 > 2 ** 19)

# ---- tempos (texto contra dados/c16_tempos.csv) -----------------
t = pd.read_csv(AQUI.parents[1] / "dados" / "c16_tempos.csv")
s = t.set_index(["experimento", "metodo", "n"])["segundos"]
confere("busca: 4,6 ms, 25 ns, 121 ns, ~185 mil vezes",
        round(s["busca", "lista_in", 10**6] * 1e3, 1) == 4.6
        and round(s["busca", "set_in", 10**6] * 1e9) == 25
        and round(s["busca", "bisect", 10**6] * 1e9) == 121
        and round(s["busca", "lista_in", 10**6]
                  / s["busca", "set_in", 10**6], -3) == 185_000)
lr = lambda m, n: s["logret", m, n]
confere("log-retornos: 62,6 ms, 3,1 ms, 20 vezes",
        round(lr("laco", 10**6) * 1e3, 1) == 62.6
        and round(lr("numpy", 10**6) * 1e3, 1) == 3.1
        and round(lr("laco", 10**6) / lr("numpy", 10**6)) == 20)
confere("n=10: NumPy 0,66 contra 0,50 microssegundo",
        round(lr("numpy", 10) * 1e6, 2) == 0.66
        and round(lr("laco", 10) * 1e6, 2) == 0.50)
confere("compreensão ganha cerca de 5% do laço (n >= 10^3)",
        all(0.03 < 1 - lr("compreensao", n) / lr("laco", n) < 0.09
            for n in (10**3, 10**4, 10**5, 10**6)))
confere("NumPy cerca de 20 vezes (n >= 10^4)",
        all(17 < lr("laco", n) / lr("numpy", n) < 22
            for n in (10**4, 10**5, 10**6)))
b = lambda m: s["bova11", m, 3967]
confere("BOVA11 negócio: 178 ms, 0,14 ms, ~1.300 vezes",
        round(b("negocio_ing") * 1e3) == 178
        and round(b("negocio") * 1e3, 2) == 0.14
        and round(b("negocio_ing") / b("negocio"), -2) == 1300)
confere("BOVA11 drawdown: 60 ms, 0,55 ms, ~110 vezes",
        round(b("dd_ingenuo") * 1e3) == 60
        and round(b("dd") * 1e3, 2) == 0.55
        and round(b("dd_ingenuo") / b("dd"), -1) == 110)
confere("ingênua do dd ~ 1/3 da do negócio; rápida do dd ~ 4x",
        round(b("negocio_ing") / b("dd_ingenuo")) == 3
        and round(b("dd") / b("negocio")) == 4)
confere("NumPy log: vantagem perto de 20 mesmo em ~4.000 preços",
        15 < lr("laco", 10**4) / lr("numpy", 10**4) < 22)
confere("BOVA11 média móvel: 4,2 ms, 0,25 ms, 0,014 ms",
        round(b("mm_ingenua") * 1e3, 1) == 4.2
        and round(b("mm") * 1e3, 2) == 0.25
        and round(b("mm_numpy") * 1e3, 3) == 0.014)
neg = t[t.experimento == "negocio"]
inc = {}
for metodo in ("ingenuo", "uma_passada"):
    q = neg[neg.metodo == metodo]
    inc[metodo] = np.polyfit(np.log(q.n), np.log(q.segundos), 1)[0]
confere(f"inclinações 2 e 1 ({inc['ingenuo']:.2f}, "
        f"{inc['uma_passada']:.2f})",
        perto(inc["ingenuo"], 2, 0.1)
        and perto(inc["uma_passada"], 1, 0.1))

# ---- gabaritos dos exercícios -----------------------------------
rng = np.random.default_rng(2026)
confere("ex. 1: 495 e 794/4096 = 19,38%",
        math.comb(12, 8) == 495
        and sum(math.comb(12, k) for k in range(8, 13)) == 794
        and round(794 / 4096 * 100, 2) == 19.38)
media6 = 6 * sum(1 / i for i in range(1, 7))
sims = []
for _ in range(100_000):
    vistos, n = set(), 0
    while len(vistos) < 6:
        vistos.add(int(rng.integers(1, 7)))
        n += 1
    sims.append(n)
confere("ex. 2: 14,7 (simulado dentro de 4 ep)",
        round(media6, 1) == 14.7
        and abs(np.mean(sims) - media6)
        < 4 * np.std(sims) / math.sqrt(len(sims)))
u = np.sort(rng.random((1_000_000, 2)), axis=1)
pedacos = np.column_stack([u[:, 0], u[:, 1] - u[:, 0], 1 - u[:, 1]])
tri = (pedacos < 0.5).all(axis=1).mean()
confere(f"ex. 3: triângulo 1/4 (simulado {tri:.4f})",
        abs(tri - 0.25) < 4 * math.sqrt(0.25 * 0.75 / 1e6))
logp = np.log(px.values)
r = np.diff(logp)
melhor, atual = 0.0, 0.0
for x in r:                                   # Kadane
    atual = max(0.0, atual + x)
    melhor = max(melhor, atual)
confere("ex. 4: Kadane = melhor negócio em log",
        perto(melhor, melhor_negocio(list(logp))[0], 1e-10))


def preco(y):
    return sum(100 / (1 + y) ** k for k in range(1, 6)) \
        + 1000 / (1 + y) ** 5


lo, hi, it = 0.0, 1.0, 0
while hi - lo >= 1e-8:
    meio = (lo + hi) / 2
    lo, hi = (meio, hi) if preco(meio) > 950 else (lo, meio)
    it += 1
confere("ex. 7: y = 11,37% com 27 iterações",
        round((lo + hi) / 2 * 100, 2) == 11.37 and it == 27)


def espera(padrao, sims=100_000, n=120):
    m = rng.integers(0, 2, size=(sims, n))
    alvo = [1 if c == "C" else 0 for c in padrao]
    ok = np.ones((sims, n - 2), dtype=bool)
    for k, a in enumerate(alvo):
        ok &= m[:, k:n - 2 + k] == a
    assert ok.any(axis=1).all()
    w = ok.argmax(axis=1) + 3
    return w.mean(), w.std() / math.sqrt(sims)


for padrao, esperado in (("CKC", 10), ("CCK", 8)):
    mu, e = espera(padrao)
    confere(f"ex. 8: {padrao} = {esperado} (simulado {mu:.3f})",
            abs(mu - esperado) < 4 * e)

print(f"\n{len(falhas)} falha(s)")
if falhas:
    raise SystemExit(1)
