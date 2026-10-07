"""Três formas de preencher um buraco; duas delas olham o futuro."""
import sys; sys.path.insert(0, "..")
from estilo import *

p = precos("^BVSP", "^GSPC").dropna(subset="^BVSP")["^GSPC"].loc["2024-11-25":"2024-12-03"]
x = np.arange(len(p)); k = int(np.where(p.isna())[0][0])
fig, ax = figura(altura=62)
ax.axvspan(k + 0.5, len(p) - 0.5, color=LILAS, zorder=0)
ax.axvline(k + 0.5, color=TINTA, lw=0.6, ls=(0, (3, 3)))
ax.plot(x[:k], p.values[:k], "-o", color=ROXO, ms=4)
ax.plot(x[k + 1:], p.values[k + 1:], "-o", color=ROXO, ms=4)
ant, dep = p.values[k - 1], p.values[k + 1]
ax.plot([k - 1, k], [ant, ant], color=VERDE, lw=1, ls="--"); ax.plot(k, ant, "s", color=VERDE, ms=5)
ax.plot([k - 1, k + 1], [ant, dep], color=OCRE, lw=1, ls="--"); ax.plot(k, (ant + dep) / 2, "D", color=OCRE, ms=4.5)
ax.plot([k, k + 1], [dep, dep], color=ROSA, lw=1, ls="--"); ax.plot(k, dep, "^", color=ROSA, ms=5)
kw = dict(fontsize=6.8, ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA, shrinkB=4, relpos=(1, 0.5)))
topo = p.max() + 4
ax.annotate("bfill: copia o valor de 29/11", (k, dep), xytext=(0.75, topo), **kw)
ax.annotate("interpolação: média de 27/11 e 29/11", (k, (ant + dep) / 2), xytext=(-0.3, topo - 8), **kw)
ax.annotate("ffill: repete o valor de 27/11", (k, ant), xytext=(1.2, p.min() - 6), **kw)
ax.text(k + 0.58, p.min() - 9, "ainda não aconteceu\nno fechamento de 28/11", fontsize=6.8, va="bottom")
ax.set_xticks(x); ax.set_xticklabels([f"{d:%d/%m}" for d in p.index])
ax.set_ylim(p.min() - 12, p.max() + 8); ax.set_xlim(-0.4, len(p) - 0.5)
ax.set_title("S&P 500 no calendário da B3: o feriado de 28/11/2024")
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
salvar(fig, "cap02/preencher")
