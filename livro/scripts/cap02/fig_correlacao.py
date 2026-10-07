"""Matriz de correlação dos retornos diários."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l11_eda import corr

ordem = ["PETR4", "VALE3", "ITUB4", "WEGE3", "Ibovespa", "S&P 500 (US$)", "S&P 500 (R$)", "Dólar"]
c = corr.loc[ordem, ordem]
fig, ax = figura(altura=82)
im = ax.imshow(c.values, cmap=DIV, vmin=-1, vmax=1)
n = len(ordem)
for i in range(n):
    for j in range(n):
        v = c.values[i, j]
        ax.text(j, i, f"{v:.2f}".replace(".", ",").replace("-", "−"), ha="center", va="center", fontsize=6.6,
                color="white" if abs(v) > 0.6 else TINTA)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(ordem, rotation=35, ha="right"); ax.set_yticklabels(ordem)
ax.tick_params(colors=TINTA, length=0)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1])
cb.outline.set_visible(False); cb.ax.tick_params(length=0)
cb.ax.set_yticklabels(["−1", "−0,5", "0", "0,5", "1"])
salvar(fig, "cap02/correlacao")
