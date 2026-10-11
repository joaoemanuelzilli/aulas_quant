"""ln(1+R) contra R: iguais perto de zero, afastam-se nas caudas."""
import sys; sys.path.insert(0, "..")
from estilo import *

R = np.linspace(-0.6, 0.8, 400)
fig, ax = figura(altura=56)
ax.plot(R, R, color=CINZA, lw=0.9, ls=(0, (4, 3)))
ax.plot(R, np.log1p(R), color=ROXO)
ax.axvspan(-0.05, 0.05, color=LILAS, zorder=0)
ax.text(0.0, -0.84, "retornos diários\ntípicos", ha="center", fontsize=6.8, color=TINTA)
ax.annotate("log-retorno\n$r=\\ln(1+R)$", (0.74, np.log1p(0.74)), xytext=(0, -22), textcoords="offset points",
            ha="center", fontsize=7.2)
ax.annotate("reta $r=R$", (0.55, 0.55), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=7.2, color=CINZA)
for x in (-0.5, 0.5):
    ax.plot([x, x], [x, np.log1p(x)], color=OCRE, lw=1)
    ax.plot([x], [np.log1p(x)], "o", color=OCRE, ms=3.5)
ax.annotate("$R=-50\\%$\n$r=-69{,}3\\%$", (-0.5, np.log1p(-0.5)), xytext=(8, -2), textcoords="offset points", fontsize=6.8, va="center")
ax.annotate("$R=+50\\%$\n$r=+40{,}5\\%$", (0.5, np.log1p(0.5)), xytext=(-4, -17), textcoords="offset points", fontsize=6.8, va="center")
ax.set_xlabel("retorno simples $R$"); ax.set_ylabel("log-retorno $r$")
pct(ax, "x"); pct(ax, "y")
ax.grid(False)
ax.axhline(0, color=CINZA_CLARO, lw=0.5, zorder=0); ax.axvline(0, color=CINZA_CLARO, lw=0.5, zorder=0)
salvar(fig, "cap01/simples_vs_log")
