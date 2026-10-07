"""Patrimônio e posição do bot de exemplo sobre a corretora simulada."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l04_bot import cor, curva, b

fig, a1 = figura(altura=56)
a1.plot(curva.index, curva / 1000, color=ROXO)
a1.axhline(100, color=CINZA, lw=0.6, ls=(0, (4, 3)))
a1.set_ylabel("capital (R$ mil)"); a1.set_yticks([100, 105, 110, 115])
stops = [(e[0], curva[e[0]] / 1000) for e in cor.extrato if e[3] == "stop"]
a1.plot([d for d, _ in stops], [v for _, v in stops], "o", color=OCRE, ms=2.6, zorder=5)
d0 = pd.Timestamp("2021-02-22")
a1.annotate("stop executado em gap,\n22/02/2021", (d0, curva[d0] / 1000), xytext=(-14, 30), textcoords="offset points", fontsize=6.8, ha="right",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA, shrinkB=3))
salvar(fig, "cap11/bot")
print(curva.iloc[-1], len(stops), curva["2021-02-19"], curva[d0], curva[d0] / curva["2021-02-19"] - 1)
