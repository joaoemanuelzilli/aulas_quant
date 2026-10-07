"""Carteiras de dois ativos para várias correlações."""
import sys; sys.path.insert(0, "..")
from estilo import *

mu_a, mu_b, s_a, s_b = 0.10, 0.15, 0.20, 0.30
w = np.linspace(0, 1, 201)
mu = w * mu_a + (1 - w) * mu_b
fig, ax = figura(altura=62)
casos = [(1, CINZA, "$\\rho=1$", (4, -9)),
         (0.3, ROXO, "$\\rho=0{,}3$", (-3, 9)),
         (-0.5, OCRE, "$\\rho=-0{,}5$", (-4, 9)),
         (-1, VERDE, "$\\rho=-1$", (5, 9))]
for rho, cor, rot, desl in casos:
    var = ((w * s_a) ** 2 + ((1 - w) * s_b) ** 2
           + 2 * w * (1 - w) * rho * s_a * s_b)
    s = np.sqrt(np.maximum(var, 0))
    ax.plot(s, mu, color=cor, lw=1.3 if rho == 0.3 else 1.0)
    k = np.argmin(np.abs(w - 0.5))
    ax.plot(s[k], mu[k], "o", color=cor, ms=3.6)
    ax.annotate(rot, (s[k], mu[k]), xytext=desl,
                textcoords="offset points", fontsize=7.2,
                ha="right" if desl[0] < 0 else "left", va="center")
ax.plot([s_a, s_b], [mu_a, mu_b], "o", color=TINTA, ms=4.2)
ax.annotate("A", (s_a, mu_a), xytext=(5, -7),
            textcoords="offset points", fontsize=7.6)
ax.annotate("B", (s_b, mu_b), xytext=(5, 2),
            textcoords="offset points", fontsize=7.6)
ax.annotate("os pontos marcam a\ncarteira 50/50", (0.05, 0.125),
            xytext=(0.012, 0.142), fontsize=6.8, color=CINZA,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.set_xlim(0, 0.33); ax.set_ylim(0.092, 0.158)
ax.set_xlabel("volatilidade da carteira")
ax.set_ylabel("retorno esperado")
pct(ax, "x"); pct(ax, "y")
salvar(fig, "cap06/dois_ativos")
