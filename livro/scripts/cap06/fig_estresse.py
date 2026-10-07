"""Perda das carteiras de hoje em quatro episódios históricos."""
import sys; sys.path.insert(0, "..")
from estilo import *

c = pd.read_csv(DADOS / "c06_estresse_historico.csv", index_col=0)
rot = {"Covid-19": "Covid-19\n23/01 a 23/03/2020",
       "Joesley Day": "Joesley Day\n18/05/2017",
       "fev-jun/2018": "Queda de 2018\n26/02 a 18/06/2018",
       "2021-2022": "Alta de juros\n07/06/2021 a 14/07/2022"}
series = [("1/N", ROXO), ("mín. var.", OCRE), ("máx. Sharpe", VERDE),
          ("Ibovespa", CINZA_CLARO)]
fig, ax = figura(altura=70)
y = np.arange(len(c))
for k, (nome, cor) in enumerate(series):
    v = c[nome].values * 100
    pos = y + (k - 1.5) * 0.2
    ax.barh(pos, v, height=0.18, color=cor, label=nome)
    for p_, v_ in zip(pos, v):
        txt = f"{v_:+.1f}".replace(".", ",").replace("-", "−") + "%"
        ax.text(v_ - 0.6 if v_ < 0 else v_ + 0.6, p_, txt, va="center",
                ha="right" if v_ < 0 else "left", fontsize=6.2)
ax.set_yticks(y, [rot[i] for i in c.index]); ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=7, labelcolor=TINTA)
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
ax.set_xlim(-56, 10)
ax.axvline(0, color=CINZA, lw=0.6)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%".replace("-", "−"))
ax.set_xlabel("retorno da carteira no episódio")
ax.legend(loc="lower left", fontsize=6.8, ncols=1)
salvar(fig, "cap06/estresse_historico")
