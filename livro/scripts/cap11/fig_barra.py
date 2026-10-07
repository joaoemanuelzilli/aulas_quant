"""A mesma barra OHLC, dois caminhos possíveis dentro do dia."""
import sys; sys.path.insert(0, "..")
from estilo import *

O, H, L, C = 10.00, 10.90, 9.40, 10.20
STOP, ALVO = 9.50, 10.80
rng = np.random.default_rng(3)

def caminho(pontos, n=120):
    xs = np.linspace(0, 1, len(pontos)); t = np.linspace(0, 1, n)
    base = np.interp(t, xs, pontos)
    ruido = rng.normal(0, 0.02, n).cumsum(); ruido -= np.linspace(ruido[0], ruido[-1], n)
    y = base + 0.6 * ruido * np.sin(np.pi * t) ** 0.5
    return t, np.clip(y, L, H)

fig, (a0, a1, a2) = figura(altura=55, ncols=3, sharey=True, gridspec_kw={"width_ratios": [0.42, 1, 1]})
a0.plot([0, 0], [L, H], color=TINTA, lw=0.8)
a0.add_patch(mpl.patches.Rectangle((-0.16, O), 0.32, C - O, fc=LILAS, ec=TINTA, lw=0.8))
for y, s in ((H, "máxima"), (L, "mínima"), (O, "abertura"), (C, "fechamento")):
    a0.text(0.24, y, s, fontsize=6.6, va="center")
a0.set_xlim(-0.5, 1.5); a0.set_xticks([]); a0.spines["bottom"].set_visible(False)
a0.set_title("A barra")
for ax, pts, tit, quem in ((a1, [O, 10.5, H, 10.3, L, 9.8, C], "Caminho A: alvo primeiro", ALVO),
                           (a2, [O, 9.7, L, 9.9, H, 10.5, C], "Caminho B: stop primeiro", STOP)):
    t, y = caminho(pts)
    y[0], y[-1] = O, C
    ax.plot(t, y, color=ROXO, lw=1)
    ax.axhline(ALVO, color=CINZA, lw=0.6, ls=(0, (4, 3))); ax.axhline(STOP, color=CINZA, lw=0.6, ls=(0, (4, 3)))
    k = np.argmax(y >= quem) if quem == ALVO else np.argmax(y <= quem)
    ax.plot([t[k]], [quem], "o", color=OCRE, ms=4.5, zorder=5)
    ax.set_title(tit); ax.set_xticks([0, 1]); ax.set_xticklabels(["abertura", "fechamento"])
a1.text(0.98, ALVO + 0.14, "venda limitada a 10,80", fontsize=6.6, ha="right", va="bottom", color=CINZA)
a2.text(0.98, 9.37, "stop de venda a 9,50", fontsize=6.6, ha="right", va="top", color=CINZA)
a0.set_ylim(9.2, 11.1); a0.set_yticks([9.5, 10.0, 10.5, 11.0]); virgula(a0, "y", 1); a0.set_ylabel("preço (R$)")
for ax in (a0, a1, a2): ax.grid(False)
salvar(fig, "cap11/barra_ambigua")
