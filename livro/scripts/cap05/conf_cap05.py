"""Confere o Capítulo 5 contra a execução dos scripts e refaz as contas.

Rode de dentro de livro/scripts/cap05:  python conf_cap05.py
1. cada bloco style=saida de capitulos/cap05.tex é trecho literal da
   saída do script impresso logo antes dele, e nenhuma linha de código
   impresso passa de 66 caracteres;
2. as fórmulas do capítulo (derivadas da LTN, gregas de Black-Scholes,
   Lagrange, KKT, problemas de entrevista) são refeitas com sympy;
3. os números citados no texto são recalculados.
Termina com erro (AssertionError) se alguma conta deixar de valer.
"""
import re
import runpy
import subprocess
import sys
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from calculo import (bs, convexidade, duration_mod, gradiente_descendente,
                     gregas, preco_ltn)

TEX = Path("../../capitulos/cap05.tex").read_text(encoding="utf8")


def perto(a, b, tol):
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


# ------------------------------------------------ 1. saídas coladas
PADRAO = re.compile(
    r"\\lstinputlisting\[[^\]]*\]\{scripts/cap05/(\w+\.py)\}\s*"
    r"\\begin\{lstlisting\}\[style=saida\]\n(.*?)\\end\{lstlisting\}",
    re.S)


def limpo(t):
    return "\n".join(l.rstrip() for l in t.strip("\n").splitlines())


blocos = PADRAO.findall(TEX)
assert len(blocos) == 9, f"esperava 9 pares código/saída, achei {len(blocos)}"
for script, colado in blocos:
    longas = [l for l in Path(script).read_text(encoding="utf8")
              .splitlines() if len(l) > 66]
    assert not longas, f"{script}: linha longa: {longas[0]}"
    r = subprocess.run([sys.executable, "-W", "error", script],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{script} falhou:\n{r.stderr}"
    assert limpo(colado) in limpo(r.stdout), f"{script}: saída difere"
    print(f"ok  {script}")

# ------------------------------------------------ 2. LTN
taxa, du = 0.1436, 532
n = du / 252
P = preco_ltn(taxa, du)
# regra oficial: expoente truncado na 14a casa, preço na 6a
exp14 = Decimal(du) / Decimal(252)
exp14 = exp14.quantize(Decimal("1e-14"), rounding=ROUND_DOWN)
fator = Decimal("1.1436") ** exp14
oficial = (Decimal(1000) / fator).quantize(Decimal("1e-6"),
                                           rounding=ROUND_DOWN)
assert str(oficial) == "753.315323", oficial
y, nn = sp.symbols("y n", positive=True)
Ps = 1000 * (1 + y) ** (-nn)
assert sp.simplify(sp.diff(Ps, y) + nn / (1 + y) * Ps) == 0
assert sp.simplify(sp.diff(Ps, y, 2) / Ps
                   - nn * (nn + 1) / (1 + y) ** 2) == 0
dP = -n * P / (1 + taxa)
perto(dP, -1390.64, 0.005)
perto(-dP * 1e-4, 0.139, 0.0005)
P300 = preco_ltn(taxa + 0.03, du)
perto(P300, 713.24, 0.005); perto(P - P300, 40.08, 0.005)
perto((P300 - P) / 0.03, -1336, 0.5)
D, C = duration_mod(taxa, du), convexidade(taxa, du)
perto(D, 1.85, 0.005); perto(C, 5.022, 0.0005)


def erro1(pb, d=du):
    dy = pb / 1e4
    ex = preco_ltn(taxa + dy, d) / preco_ltn(taxa, d) - 1
    return -duration_mod(taxa, d) * dy - ex, ex


perto(erro1(100)[0] * 100, -0.025, 0.0005)            # p.p.
perto(-erro1(100)[0] * P, 0.19, 0.005)                # R$
perto(-erro1(300)[0] * 100, 0.22, 0.005)
perto(-erro1(300)[0] * P, 1.64, 0.005)
for pb in (-300, 300):
    dy = pb / 1e4
    e2 = (-D * dy + 0.5 * C * dy**2) - erro1(pb)[1]
    assert abs(e2) < 1e-4                             # < 0,01 p.p.
perto(-erro1(300, 2520)[0] * 100, 3.4, 0.05)
perto(-erro1(-300, 2520)[0] * 100, 4.2, 0.05)          # queda de 300 pb
perto(duration_mod(taxa, 2520), 8.74, 0.005)
perto(convexidade(taxa, 2520), 84.1, 0.05)
# Taylor de ln(1+R) na queda de 29,7%
perto(np.log(1 - 0.297), -0.352, 0.0005)
perto(-0.297 - 0.297**2 / 2, -0.341, 0.0005)

# CDI: choque de 2021-2022 e taxa do fim de 2025
bcb = pd.read_csv("../../dados/c02_bcb.csv", index_col=0,
                  parse_dates=True)
aa = (1 + bcb["cdi"] / 100) ** 252 - 1
perto(aa.loc["2021-02"].mean(), 0.0190, 0.00005)
perto(aa.loc["2022-08"].max(), 0.1365, 0.00005)
assert aa.loc["2022-08"].max() - aa.loc["2021-02"].mean() > 0.11
cdi_fim = bcb["cdi"].iloc[-1] / 100
assert bcb.index[-1] == pd.Timestamp("2025-12-31")
perto((1 + cdi_fim) ** 252 - 1, 0.1490, 0.00005)
perto(252 * np.log1p(cdi_fim), 0.139, 0.0005)
print("ok  LTN, duration, convexidade e CDI")

# ------------------------------------------------ 3. parciais
S2 = np.array([[0.04, 0.018], [0.018, 0.09]])
g = 2 * S2 @ np.array([0.5, 0.5])
perto(g[0], 0.058, 1e-12); perto(g[1], 0.108, 1e-12)
w1, w2, s1, s2, s12 = sp.symbols("w1 w2 s1 s2 s12")
f = w1**2 * s1**2 + 2 * w1 * w2 * s12 + w2**2 * s2**2
H = sp.hessian(f, (w1, w2))
assert H == sp.Matrix([[2 * s1**2, 2 * s12], [2 * s12, 2 * s2**2]])

# ------------------------------------------------ 4. regressão e GD
ns = runpy.run_path("l03_gradiente.py")
x, yv = ns["x"], ns["y"]
b = np.cov(x, yv, ddof=0)[0, 1] / x.var()
perto(b, ns["b_mqo"], 1e-10)
perto(yv.mean() - b * x.mean(), ns["a_mqo"], 1e-10)
Hl = 2 * np.array([[1, x.mean()], [x.mean(), (x**2).mean()]])
perto(Hl[1, 1], 1.31, 0.005); perto(Hl[0, 1], -0.08, 0.005)
lmax = np.linalg.eigvalsh(Hl).max()
perto(lmax, 2.009, 0.0005); perto(2 / lmax, 0.995, 0.0005)
# armadilha: retornos em fração
xf, yf = x / 100, yv / 100
Hf = 2 * np.array([[1, xf.mean()], [xf.mean(), (xf**2).mean()]])
perto(Hf[1, 1], 0.00013, 0.000005)
lmin = np.linalg.eigvalsh(Hf).min()
eta = 0.95 * 2 / np.linalg.eigvalsh(Hf).max()
passos = np.log(100) / -np.log(1 - eta * lmin)       # erro / 100
assert 10_000 < passos < 100_000, passos
print(f"ok  regressão e gradiente ({passos:.0f} passos em fração)")

# ------------------------------------------------ 5. Lagrange e KKT
lam = sp.symbols("lam")
L = (0.04 * w1**2 + 2 * 0.018 * w1 * w2 + 0.09 * w2**2
     - lam * (w1 + w2 - 1))
sol = sp.solve([sp.diff(L, v) for v in (w1, w2, lam)], (w1, w2, lam))
perto(float(sol[w1]), 72 / 94, 1e-12)
perto(float(sol[lam]), 0.0697, 0.00005)
vmin = float(sol[lam]) / 2
perto(vmin, 0.0349, 0.00005); perto(np.sqrt(vmin), 0.1867, 0.00005)
perto(float(sol[lam]) * 0.01, 0.0007, 0.00001)
S8 = np.array([[0.04, 0.048], [0.048, 0.09]])
wl = np.linalg.solve(S8, np.ones(2)); wl = wl / wl.sum()
perto(wl[0], 1.235, 0.0005)
gk = 2 * S8 @ np.array([1.0, 0.0])
perto(gk[0], 0.080, 1e-12); perto(gk[1] - gk[0], 0.016, 1e-12)
assert 0.8 > 0.20 / 0.30 > 0.3     # KKT ativa só acima de 2/3
print("ok  Lagrange e KKT")

# ------------------------------------------------ 6. Black-Scholes
Ss, Ks, Ts, rs, vs = sp.symbols("S K T r sigma", positive=True)
d1 = (sp.log(Ss / Ks) + (rs + vs**2 / 2) * Ts) / (vs * sp.sqrt(Ts))
d2 = d1 - vs * sp.sqrt(Ts)
Nc = lambda u: (1 + sp.erf(u / sp.sqrt(2))) / 2
Nd = lambda u: sp.exp(-u**2 / 2) / sp.sqrt(2 * sp.pi)
Cs = Ss * Nc(d1) - Ks * sp.exp(-rs * Ts) * Nc(d2)
formulas = {
    "delta": (sp.diff(Cs, Ss), Nc(d1)),
    "gama": (sp.diff(Cs, Ss, 2), Nd(d1) / (Ss * vs * sp.sqrt(Ts))),
    "vega": (sp.diff(Cs, vs), Ss * sp.sqrt(Ts) * Nd(d1)),
    "theta": (-sp.diff(Cs, Ts), -Ss * Nd(d1) * vs / (2 * sp.sqrt(Ts))
              - rs * Ks * sp.exp(-rs * Ts) * Nc(d2)),
    "rho": (sp.diff(Cs, rs), Ks * Ts * sp.exp(-rs * Ts) * Nc(d2)),
}
rng = np.random.default_rng(1)
for _ in range(5):                      # cinco pontos sorteados
    pt = {Ss: rng.uniform(10, 50), Ks: rng.uniform(10, 50),
          Ts: rng.uniform(0.05, 2), rs: rng.uniform(0, 0.2),
          vs: rng.uniform(0.1, 0.6)}
    for nome, (deriv, form) in formulas.items():
        a_, b_ = float(deriv.subs(pt)), float(form.subs(pt))
        perto(a_, b_, 1e-9 * max(1, abs(b_)))
p = dict(S=30.0, K=30.0, T=0.25, r=0.14, sigma=0.30)
C0, P0 = bs(**p), bs(**p, tipo="venda")
perto(C0 / 30, 0.078, 0.0005)
gr = gregas(**p)
tabela = {"delta": (0.621, 0.0005), "gama": (0.0845, 0.00005),
          "vega": (5.71, 0.005), "theta": (-5.71, 0.005),
          "rho": (4.08, 0.005)}
for k, (v, tol) in tabela.items():
    perto(gr[k], v, tol)
perto(gr["vega"] / 100, 0.057, 0.0005)
perto(gr["rho"] / 100, 0.041, 0.0005)
perto(gr["theta"] / 252, -0.023, 0.0005)
# delta-gama para +-5%
for dS, ex_, o1_, o2_ in [(1.5, 1.021, 0.932, 1.027)]:
    ex = bs(30 + dS, 30, 0.25, 0.14, 0.3) - C0
    perto(ex, ex_, 0.0005); perto(gr["delta"] * dS, o1_, 0.0005)
    perto(gr["delta"] * dS + 0.5 * gr["gama"] * dS**2, o2_, 0.0005)
# Monte Carlo do preço: distância à fórmula e erro-padrão
z = np.random.default_rng(42).standard_normal(1_000_000)
ST = 30 * np.exp((0.14 - 0.045) * 0.25 + 0.3 * 0.5 * z)
pago = np.exp(-0.035) * np.maximum(ST - 30, 0)
ep = pago.std(ddof=1) / np.sqrt(len(z))
perto(abs(pago.mean() - C0), 0.0005, 0.00005)
perto(abs(pago.mean() - C0) / ep, 1 / 6, 0.02)
# gama por Monte Carlo com passos diferentes
g7 = runpy.run_path("l07_gregas.py")
for h, ep_ in [(0.03, 0.0014), (3.0, 0.0001)]:
    cen = g7["dif"](g7["mc"], "S", h, 2)
    perto(cen.std(ddof=1) / np.sqrt(len(cen)), ep_, 0.00005)
vies = g7["dif"](bs, "S", 3.0, 2) / gr["gama"] - 1
perto(vies, -0.026, 0.0005)
print("ok  Black-Scholes, gregas e Monte Carlo")

# ------------------------------------------------ 7. hedge
h8 = runpy.run_path("l08_hedge.py")
pl5, liq5 = h8["vende_e_protege"](5)
pl1, liq1 = h8["vende_e_protege"](1)
perto(pl5.std() / pl1.std(), 2.1, 0.05)
perto(liq1.mean(), -0.048, 0.0005); perto(liq5.mean(), -0.031, 0.0005)
pl0, _ = h8["vende_e_protege"](0)                    # sem proteção
ST8 = h8["S"][:, -1]
perto((ST8 <= 30).mean(), 0.40, 0.005)        # fica com o prêmio
perto((pl0 < -1).mean(), 0.24, 0.005)         # perde mais que recebeu
assert -6 < np.quantile(pl0, 0.01) < -5       # pior 1%: mais de 5x
perto(pl0.max(), np.exp(0.14 * 63 / 252), 1e-9)   # prêmio com juros
perto(pl0.max(), 1.04, 0.005)
perto(pl0.min(), -12, 0.5)                    # perto de doze vezes
assert pl0.min() < -6                         # eixo da figura para em -6
print("ok  hedge delta")

# ------------------------------------------------ 8. entrevistas
xs = sp.symbols("x", positive=True)
crit = sp.solve(sp.diff(sp.log(xs) / xs, xs), xs)
assert crit == [sp.E]
perto(float(sp.E ** (1 / sp.E)), 1.4447, 0.00005)
assert sp.E ** sp.pi > sp.pi ** sp.E
u, th = sp.symbols("u theta", positive=True)
I2 = sp.integrate(sp.integrate(sp.exp(-u**2 / 2) * u, (u, 0, sp.oo)),
                  (th, 0, 2 * sp.pi))
assert sp.simplify(I2 - 2 * sp.pi) == 0
I = sp.integrate(sp.exp(-xs**2 / 2), (xs, 0, sp.oo)) * 2
assert sp.simplify(I - sp.sqrt(2 * sp.pi)) == 0
print("ok  problemas de entrevista")
print("todas as contas do capítulo conferem")
