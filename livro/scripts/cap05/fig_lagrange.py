"""Lagrange no plano dos pesos e KKT com pesos não negativos."""
import sys; sys.path.insert(0, "..")
from estilo import *


def cov(rho, s_a=0.20, s_b=0.30):
    return np.array([[s_a**2, rho * s_a * s_b],
                     [rho * s_a * s_b, s_b**2]])


fig, (a1, a2) = figura(altura=66, ncols=2)
S = cov(0.3)
g1, g2 = np.meshgrid(np.linspace(-0.2, 1.3, 201),
                     np.linspace(-0.4, 1.1, 201))
V = S[0, 0] * g1**2 + 2 * S[0, 1] * g1 * g2 + S[1, 1] * g2**2
w = np.linalg.solve(S, np.ones(2)); w = w / w.sum()
v0 = w @ S @ w
a1.contour(g1, g2, V, levels=[0.01, 0.02, v0, 0.06, 0.09],
           colors=CINZA, linewidths=0.6)
t = np.linspace(-0.2, 1.3, 2)
a1.plot(t, 1 - t, color=OCRE, lw=1.2)
a1.annotate("$w_1 + w_2 = 1$", (0.1, 0.9), fontsize=7.2,
            xytext=(5, 3), textcoords="offset points")
gf = 2 * S @ w
a1.annotate("", w + 6 * gf, w, arrowprops=dict(arrowstyle="-|>",
            color=ROXO, lw=1.0))
a1.annotate("", w + 0.15, w, arrowprops=dict(arrowstyle="-|>", color=OCRE, lw=1.0))
a1.plot(*w, "o", color=TINTA, ms=3.6, zorder=5)
a1.annotate("ótimo\n(0,766; 0,234)", w, xytext=(-9, -6),
            textcoords="offset points", fontsize=7.2, ha="right")
a1.annotate("$\\nabla f$", w + 6 * gf, xytext=(3, -3),
            textcoords="offset points", fontsize=7.6)
a1.annotate("$\\nabla g$", w + 0.15, xytext=(3, -10),
            textcoords="offset points", fontsize=7.6)
a1.set_aspect("equal"); a1.grid(False)
a1.set_xlim(-0.2, 1.3); a1.set_ylim(-0.4, 1.1)
a1.set_xlabel("$w_1$ (ativo A)"); a1.set_ylabel("$w_2$ (ativo B)")
virgula(a1, "x"); virgula(a1, "y")
a1.set_title("Lagrange, $\\rho = 0{,}3$", fontsize=7.6)

wa = np.linspace(-0.2, 1.5, 341)
for rho, cor, rot in [(0.3, CINZA, "$\\rho = 0{,}3$"),
                      (0.8, ROXO, "$\\rho = 0{,}8$")]:
    S = cov(rho)
    v = (S[0, 0] * wa**2 + 2 * S[0, 1] * wa * (1 - wa)
         + S[1, 1] * (1 - wa)**2)
    a2.plot(wa, np.sqrt(v), color=cor, lw=1.2)
    k = np.argmin(v)
    a2.plot(wa[k], np.sqrt(v[k]), "o", color=cor, ms=3.2)
    a2.annotate(rot, (wa[k], np.sqrt(v[k])), xytext=(0, -11),
                textcoords="offset points", fontsize=7.2,
                ha="center")
a2.axvspan(1, 1.5, color=CINZA_CLARO, alpha=0.5, lw=0)
a2.axvspan(-0.2, 0, color=CINZA_CLARO, alpha=0.5, lw=0)
a2.annotate("proibido\n($w_B < 0$)", (1.25, 0.31), fontsize=7.2,
            ha="center")
S = cov(0.8)
a2.plot(1, 0.2, "o", color=TINTA, ms=3.6, zorder=5)
a2.annotate("ótimo com $w_B \\geq 0$", (1, 0.2),
            xytext=(0.62, 0.262), fontsize=7.2, ha="right",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a2.set_xlim(-0.2, 1.5); a2.set_ylim(0.16, 0.33)
a2.set_xlabel("$w_A$ (com $w_B = 1 - w_A$)")
a2.set_ylabel("volatilidade da carteira")
pct(a2, "y"); virgula(a2, "x")
a2.set_title("KKT: pesos não negativos", fontsize=7.6)
salvar(fig, "cap05/lagrange_kkt")
