"""Estatística t do retorno ativo quando se retira uma ação."""
import sys; sys.path.insert(0, "..")
from estilo import *
from projeto import *

px, ret, cdi = carregar()


def t_ativo(acoes):
    r, _, _ = simular(alvo_momentum(px, 12, 2, acoes), ret, TESTE)
    b, _, _ = simular(alvo_fixo(px, acoes), ret, TESTE)
    return ativo_mensal(r, b)[2]


cheio = t_ativo(ACOES)
sem = pd.Series({a[:-3]: t_ativo([x for x in ACOES if x != a])
                 for a in ACOES}).sort_values()
print("universo completo:", round(cheio, 2))
print(sem.round(2).to_string())

fig, ax = figura(altura=45)
ax.barh(["sem " + a for a in sem.index], sem, color=ROXO, height=0.62)
for i, v in enumerate(sem):
    ax.text(v - 0.04, i, f"{v:.2f}".replace(".", ","), va="center",
            ha="right", fontsize=7, color="white")
ax.axvline(cheio, color=TINTA, lw=0.8, ls=(0, (4, 2)), zorder=0.5)
ax.axvline(2, color=CINZA, lw=0.8)
ax.text(cheio - 0.03, 6.62, "sete ações", ha="right", fontsize=7)
ax.text(1.97, 6.62, "t = 2", ha="right", fontsize=7)
ax.set_xlim(0, 2.75)
ax.set_ylim(-0.6, 7.0)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
virgula(ax, "x")
ax.set_xlabel("estatística t do retorno ativo mensal, teste")
salvar(fig, "cap12/universo")
m = px["MGLU3.SA"]
print("MGLU3, fim de 2015 a fim de 2019: x",
      round(m.loc["2019-12-30"] / m.loc["2015-12-30"]))
