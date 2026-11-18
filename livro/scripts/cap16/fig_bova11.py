"""Figura do Capítulo 16: melhor negócio e drawdown máximo de BOVA11."""
import sys

sys.path.insert(0, "..")
from estilo import *  # noqa: E402,F403
from algoritmos import melhor_negocio  # noqa: E402

px = precos("BOVA11.SA")["BOVA11.SA"].dropna()
_, i, j = melhor_negocio(list(np.log(px)))
compra, venda = px.index[i], px.index[j]
dd = px / px.cummax() - 1
vale = dd.idxmin()
pico = px[:vale].idxmax()

fig, ax = figura(altura=62)
ax.semilogy(px.index, px, color=ROXO, lw=1.0)
ax.axvspan(pico, vale, color=CINZA_CLARO, alpha=0.45, lw=0)
ax.plot([pico, vale], [px[pico], px[vale]], color=OCRE, lw=1.2,
        ls="--")
ax.plot([compra, venda], [px[compra], px[venda]], color=VERDE,
        lw=1.2, ls="--")
for d, txt, dx, dy, ha in (
        (pico, f"pico {pico:%d/%m/%Y}\nR$ {px[pico]:.2f}", 4, 8, "left"),
        (vale, f"vale e compra {vale:%d/%m/%Y}\nR$ {px[vale]:.2f}",
         6, -12, "left"),
        (venda, f"venda {venda:%d/%m/%Y}\nR$ {px[venda]:.2f}",
         -6, 6, "right")):
    ax.plot(d, px[d], "o", color=TINTA, ms=3)
    ax.annotate(txt.replace(".", ","), (d, px[d]), xytext=(dx, dy),
                textcoords="offset points", ha=ha, fontsize=6.8,
                color=TINTA)
rotulo = f"drawdown máximo\n{dd.min():.1%}".replace(".", ",")
ax.text(pico + (vale - pico) / 2, 125, rotulo.replace("-", "\u2212"),
        ha="center", fontsize=6.8, color=CINZA)
ax.set_yticks([40, 60, 80, 100, 150])
ax.set_yticklabels(["40", "60", "80", "100", "150"])
ax.minorticks_off()
ax.set_ylabel("preço ajustado (R$, escala log)")
ax.set_ylim(30, 190)
salvar(fig, "cap16/bova11")
