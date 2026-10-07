"""Sharpe por par de janelas, dentro e fora da amostra."""
import sys; sys.path.insert(0, "..")
from estilo import *
from otimizacao import grade, RAPIDAS, LENTAS

g = grade()
melhor = g.loc[g["dentro"].idxmax()]
lim = 0.8
fig, eixos = figura(altura=62, ncols=2, sharey=True)
titulos = {"dentro": "Dentro da amostra (2011 a 2018)",
           "fora": "Fora da amostra (2019 a 2025)"}
for ax, (col, titulo) in zip(eixos, titulos.items()):
    m = g.pivot(index="lenta", columns="rapida", values=col).reindex(index=LENTAS, columns=RAPIDAS)
    im = ax.imshow(m.values, origin="lower", cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(RAPIDAS))[1::2]); ax.set_xticklabels(RAPIDAS[1::2])
    ax.set_yticks(range(len(LENTAS))[1::2]); ax.set_yticklabels(LENTAS[1::2])
    ax.set_title(titulo, fontsize=7.6); ax.set_xlabel("média rápida (dias)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    i, j = LENTAS.index(melhor["lenta"]), RAPIDAS.index(melhor["rapida"])
    ax.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=TINTA, lw=1.2))
    ax.text(j + 0.75, i, f"{melhor[col]:.2f}".replace(".", ","), fontsize=7, va="center")
eixos[0].set_ylabel("média lenta (dias)")
cb = fig.colorbar(im, ax=eixos, shrink=0.85, pad=0.02, aspect=22)
cb.outline.set_visible(False); cb.ax.tick_params(length=0)
cb.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])
cb.set_ticklabels(["−0,8", "−0,4", "0", "0,4", "0,8"])
cb.set_label("índice de Sharpe")
salvar(fig, "cap07/mapa_calor")
print(melhor, g["dentro"].min(), g["dentro"].max(), g["fora"].min(), g["fora"].max())
