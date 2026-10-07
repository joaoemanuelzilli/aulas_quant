"""Anatomia de uma figura do Matplotlib: os nomes dos objetos."""
import sys; sys.path.insert(0, "..")
from estilo import *

petr = precos("PETR4.SA", inicio="2024", fim="2024")["PETR4.SA"].dropna()
ret = petr.pct_change().dropna()
mpl.rcParams["figure.constrained_layout.use"] = False
fig = plt.figure(figsize=(L_TEXTO, 78 * MM))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], left=0.3, right=0.74, top=0.84, bottom=0.13, hspace=0.42)
a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharex=a1)
a1.plot(petr.index, petr, color=ROXO)
a1.set_title("PETR4 em 2024 (R$)")
a2.bar(ret.index, ret * 100, color=CINZA, width=1.6)
a2.axhline(0, color=TINTA, lw=0.6)
a2.set_title("Retorno diário (%)")
a2.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
a2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m/%Y"))
plt.setp(a1.get_xticklabels(), visible=False)
pior = ret.idxmin()
a2.annotate(f"{ret[pior]:.1%}".replace(".", ","), (pior, ret[pior] * 100), xytext=(5, -6),
            textcoords="offset points", fontsize=6.4)
a2.set_ylim(-14, 9)

# moldura da Figure e caixas dos Axes
fig.add_artist(mpl.patches.Rectangle((0.005, 0.005), 0.99, 0.99, fill=False, ec=TINTA, lw=0.7,
                                     transform=fig.transFigure))
for a in (a1, a2):
    b = a.get_position()
    fig.add_artist(mpl.patches.Rectangle((b.x0, b.y0), b.width, b.height, fill=False, ec=ROXO, lw=0.6,
                                         ls=(0, (3, 2)), transform=fig.transFigure))

kw = dict(xycoords="figure fraction", textcoords="figure fraction", fontsize=6.9, color=TINTA,
          arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA, shrinkA=2, shrinkB=1))
b1, b2 = a1.get_position(), a2.get_position()


def nota(texto, xy, xytext, ha):
    fig.text(*xytext, texto, fontsize=6.9, ha=ha, va="center", linespacing=1.25)
    dx = 0.008 if ha == "right" else -0.008
    fig.add_artist(mpl.lines.Line2D([xytext[0] + dx, xy[0]], [xytext[1], xy[1]], color=CINZA, lw=0.5,
                                    transform=fig.transFigure))


fig.text(0.02, 0.955, "Figure: a folha inteira", fontsize=6.9, weight="semibold", va="center")
nota("Axes: um gráfico\ncom seus dois eixos", (b1.x0, b1.y1 - 0.06), (0.235, b1.y1 - 0.06), "right")
nota("eixo y (YAxis):\nmarcas e rótulos", (b1.x0 - 0.035, b1.y0 + 0.12), (0.2, b1.y0 + 0.06), "right")
nota("outro Axes\nna mesma Figure", (b2.x0, b2.y0 + 0.1), (0.235, b2.y0 + 0.1), "right")
nota("eixo x (XAxis),\ncompartilhado", (b2.x1 + 0.005, b2.y0 - 0.05), (0.775, b2.y0 - 0.07), "left")
nota("título do Axes\nax.set_title()", (b1.x0 + 0.27, b1.y1 + 0.04), (0.775, b1.y1 + 0.06), "left")
nota("linha (Line2D)\nax.plot()", (b1.x1 - 0.06, b1.y0 + 0.19), (0.775, b1.y0 + 0.19), "left")
nota("grade\nax.grid()", (b1.x1, b1.y0 + 0.075), (0.775, b1.y0 + 0.05), "left")
nota("barras\nax.bar()", (b2.x1 - 0.03, b2.y0 + 0.14), (0.775, b2.y0 + 0.17), "left")
nota("anotação\nax.annotate()", (b2.x0 + 0.2, b2.y0 + 0.03), (0.775, b2.y0 + 0.045), "left")
salvar(fig, "cap02/anatomia")
