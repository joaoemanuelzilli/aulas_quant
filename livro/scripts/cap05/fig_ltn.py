"""Preço do prefixado em função da taxa, com a reta tangente."""
import sys; sys.path.insert(0, "..")
from estilo import *
from calculo import preco_ltn

du, y0 = 532, 0.1436
y = np.linspace(0.0, 0.30, 301)
P0 = preco_ltn(y0, du)
d = -(du / 252) * P0 / (1 + y0)               # P'(y0)
fig, ax = figura(altura=60)
ax.plot(y, preco_ltn(y, du), color=ROXO, lw=1.4)
ax.plot(y, P0 + d * (y - y0), color=CINZA, lw=0.9, ls="--")
y1 = y0 + 0.03
P1 = preco_ltn(y1, du)
ax.plot([y0, y1], [P0, P1], color=OCRE, lw=1.1)
ax.plot([y0, y1], [P0, P1], "o", color=TINTA, ms=3.4)
ax.annotate("preço exato", (0.27, preco_ltn(0.27, du)),
            xytext=(0, 8), textcoords="offset points", fontsize=7.2,
            ha="center")
ax.annotate("reta tangente em 14,36%\n(inclinação $P'$)",
            (0.06, P0 + d * (0.06 - y0)), xytext=(0.012, 640),
            fontsize=7.2, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.annotate("secante até +300 pb", ((y0 + y1) / 2, (P0 + P1) / 2),
            xytext=(8, 10), textcoords="offset points", fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.annotate(f"R$ {P0:.2f}".replace(".", ","), (y0, P0),
            xytext=(-6, -10), textcoords="offset points",
            fontsize=7.2, ha="right")
ax.set_xlim(0, 0.30); ax.set_ylim(560, 1010)
ax.set_xlabel("taxa ao ano")
ax.set_ylabel("preço (R$)")
pct(ax, "x")
salvar(fig, "cap05/ltn_tangente")
