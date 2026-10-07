"""Assinaturas de um AR(1) e de um MA(1) na ACF e na PACF."""
import sys; sys.path.insert(0, "..")
from estilo import *
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf

T, K = 1000, 12
np.random.seed(11)
procs = {"AR(1), $\\phi = 0{,}7$": ArmaProcess([1, -0.7], [1]),
         "MA(1), $\\theta = 0{,}7$": ArmaProcess([1], [1, 0.7])}
fig, axs = figura(altura=82, nrows=2, ncols=2, sharex=True, sharey=True)
k = np.arange(1, K + 1)
banda = 1.96 / np.sqrt(T)
for i, (nome, pr) in enumerate(procs.items()):
    y = pr.generate_sample(T)
    am, pm = acf(y, nlags=K)[1:], pacf(y, nlags=K, method="ywm")[1:]
    at, pt = pr.acf(K + 1)[1:], pr.pacf(K + 1)[1:]
    print(nome, "| ACF teórica:", at[:4].round(3), "| PACF teórica:", pt[:4].round(3))
    print("   amostral ACF:", am[:4].round(3), "PACF:", pm[:4].round(3))
    for ax, am_, te_, rot in ((axs[i, 0], am, at, "ACF"), (axs[i, 1], pm, pt, "PACF")):
        ax.axhspan(-banda, banda, color=CINZA_CLARO, zorder=0)
        ax.bar(k, am_, width=0.55, color=ROXO)
        ax.plot(k, te_, "o", color=TINTA, ms=2.6, zorder=3)
        ax.axhline(0, color=CINZA, lw=0.6)
        ax.set_title(f"{rot} de um {nome}", fontsize=7.8)
        ax.set_ylim(-0.55, 0.85)
        ax.set_yticks([-0.5, 0, 0.5]); virgula(ax)
for ax in axs[1]:
    ax.set_xlabel("defasagem $k$"); ax.set_xticks([1, 3, 5, 7, 9, 11])
axs[0, 1].plot([], [], "o", color=TINTA, ms=2.6, label="valor teórico")
axs[0, 1].bar([0], [0], color=ROXO, label="estimativa em 1\u2009000 observações")
fig.legend(*axs[0, 1].get_legend_handles_labels(), loc="outside lower center", ncols=2, fontsize=6.8)
axs[0, 0].set_xlim(0.3, K + 0.7)
salvar(fig, "cap04/assinaturas")
