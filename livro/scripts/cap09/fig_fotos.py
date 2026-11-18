"""A curva do WTI (contratos 1 a 4) em quatro datas."""
import sys; sys.path.insert(0, "..")
from estilo import *
from futuros import ler

f = ler(["cl1", "cl2", "cl3", "cl4"]).dropna()
datas = [("19/12/2008", "contango"), ("21/04/2020", "contango extremo"),
         ("08/03/2022", "backwardation"), ("05/04/2024", "último dado da EIA")]
fig, axs = figura(altura=46, largura="larga", ncols=4)
for ax, (d, nome) in zip(axs, datas):
    t = pd.Timestamp(pd.to_datetime(d, dayfirst=True))
    y = f.loc[t].to_numpy()
    cor = ROXO if y[-1] > y[0] else OCRE
    ax.plot([1, 2, 3, 4], y, "-o", color=cor, ms=3.5)
    ax.set_title(d, fontsize=8)
    ax.text(0.03, 0.04, nome, transform=ax.transAxes, fontsize=6.8, color=CINZA)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.45, 4.55)
    lo, hi = y.min(), y.max(); pad = (hi - lo) * 0.35
    ax.set_ylim(lo - pad * 1.6, hi + pad)
    virgula(ax, casas=0)
    for k, v in enumerate(y, 1):
        ax.annotate(f"{v:.2f}".replace(".", ","), (k, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=6.2)
axs[0].set_ylabel("US$ por barril")
for ax in axs: ax.set_xlabel("contrato")
salvar(fig, "cap09/fotos")
