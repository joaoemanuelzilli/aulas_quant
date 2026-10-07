"""Pesos ótimos estimados em duas janelas de cinco anos."""
import sys; sys.path.insert(0, "..")
from estilo import *

t = pd.read_csv(DADOS / "c06_duas_janelas.csv", index_col=0)
fig, eixos = figura(altura=56, ncols=2, sharey=True)
y = np.arange(len(t))
for ax, regra, titulo in zip(eixos, ["MS", "MV"],
                             ["Máximo Sharpe", "Mínima variância"]):
    ax.barh(y - 0.2, t[regra + " 15-19"], height=0.38, color=ROXO,
            label="estimado em 2015–2019")
    ax.barh(y + 0.2, t[regra + " 20-24"], height=0.38, color=OCRE,
            label="estimado em 2020–2024")
    ax.set_title(titulo)
    ax.set_yticks(y, t.index)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlim(0, 62)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("peso na carteira")
eixos[0].invert_yaxis()          # eixo y é compartilhado
eixos[1].legend(loc="upper right", fontsize=6.8)
salvar(fig, "cap06/pesos_janelas")
