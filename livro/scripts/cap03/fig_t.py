"""Normal contra t de Student padronizada: centro e cauda."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy import stats

x = np.linspace(-5, 5, 600)
k = np.linspace(0, 8, 400)
fig, (a1, a2) = figura(altura=52, ncols=2)
a1.plot(x, stats.norm.pdf(x), color=CINZA, lw=1.1)
a2.plot(k, stats.norm.sf(k), color=CINZA, lw=1.1)
for nu, cor in [(3, ROXO), (5, OCRE), (10, VERDE)]:
    c = np.sqrt(nu / (nu - 2))          # T = c X, com Var(X) = 1
    a1.plot(x, c * stats.t.pdf(c * x, nu), color=cor)
    a2.plot(k, stats.t.sf(c * k, nu), color=cor)
    rotulo_final(a2, 8, stats.t.sf(c * 8, nu), f"$\\nu={nu}$", cor)
a1.set_title("Densidade (variância 1)")
a1.set_xlabel("desvios-padrão")
a1.set_xlim(-4.2, 4.2)
virgula(a1)
a1.annotate("normal", (0.95, stats.norm.pdf(0.95)), xytext=(1.9, 0.36), fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
c3 = np.sqrt(3)
a1.annotate("$t$ com $\\nu=3$", (0.18, c3 * stats.t.pdf(c3 * 0.18, 3)), xytext=(1.3, 0.56), fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a2.set_yscale("log")
a2.set_ylim(1e-10, 1)
a2.set_xlim(0, 9.6)
a2.spines["bottom"].set_bounds(0, 8)
a2.set_xticks([0, 2, 4, 6, 8])
a2.set_yticks([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
a2.minorticks_off()
a2.set_title("Probabilidade de cauda $P(X>k)$")
a2.set_xlabel("$k$ (desvios-padrão)")
a2.annotate("normal", (5.6, stats.norm.sf(5.6)), xytext=(6.2, 3e-7), fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
salvar(fig, "cap03/t_student")
for nu in (3, 5, 10):
    c = np.sqrt(nu / (nu - 2))
    print(nu, "densidade em 0:", c * stats.t.pdf(0, nu), "normal:", stats.norm.pdf(0))
