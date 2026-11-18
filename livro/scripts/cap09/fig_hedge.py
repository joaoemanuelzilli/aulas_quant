"""Retorno mensal à vista contra o do futuro de WTI rolado."""
import sys; sys.path.insert(0, "..")
from estilo import *
from futuros import fim_de_mes
from l05_hedge import razao

m = fim_de_mes(["wti", "brent", "cl1", "cl2"], "cl")
rf = (m.cl1 / m.cl2.shift() - 1).dropna()
fig, axs = figura(altura=62, ncols=2, sharey=True)
for ax, (vista, tit) in zip(axs, [("wti", "WTI à vista"), ("brent", "Brent à vista")]):
    r = (m[vista] / m[vista].shift() - 1).dropna()
    for per, cor in [(slice("2008", "2015"), ROXO), (slice("2016", "2024"), OCRE)]:
        ax.plot(rf[per], r[per], "o", ms=2.6, color=cor, alpha=0.75, mec="none")
    h = razao(r["2008":"2015"], rf["2008":"2015"])
    x = np.array([-0.55, 0.6])
    ax.plot(x, h * x + (r["2008":"2015"].mean() - h * rf["2008":"2015"].mean()), color=TINTA, lw=0.9)
    ax.text(0.04, 0.93, f"{tit}\n$h^*$ do treino = {h:.2f}".replace(".", ","),
            transform=ax.transAxes, fontsize=7.2, va="top")
    ax.set_xlabel("retorno do futuro de WTI no mês")
    ax.set_xlim(-0.6, 0.65); ax.set_ylim(-0.75, 0.75)
    pct(ax, "x"); pct(ax)
    ax.grid(axis="x")
axs[0].set_ylabel("retorno à vista no mês")
axs[1].text(0.38, -0.5, "2008–2015", color=TINTA, fontsize=7)
axs[1].plot([0.33], [-0.48], "o", ms=3, color=ROXO)
axs[1].text(0.38, -0.62, "2016–2024", color=TINTA, fontsize=7)
axs[1].plot([0.33], [-0.60], "o", ms=3, color=OCRE)
salvar(fig, "cap09/hedge")
