"""Distribuição do resultado de quem vende a opção."""
import runpy
import sys; sys.path.insert(0, "..")
from estilo import *

ns = runpy.run_path("l08_hedge.py")
f = ns["vende_e_protege"]
fig, (a1, a2) = figura(altura=58, ncols=2)
nu, _ = f(0)
a1.hist(nu, bins=np.linspace(-8, 1.2, 93), color=CINZA, lw=0)
a1.set_title("Sem proteção", fontsize=7.6)
pct(a1, "x")
a1.set_xlim(-6, 1.2)
bins = np.linspace(-0.8, 0.6, 71)
for k, cor, rot, pos in [(5, OCRE, "a cada 5 dias", (-0.75, 0.3)),
                         (1, ROXO, "todo dia", (0.09, 0.8))]:
    pl, _ = f(k)
    a2.hist(pl, bins=bins, color=cor, alpha=0.55, lw=0,
            density=True)
    a2.annotate(rot, pos, xycoords=("data", "axes fraction"),
                fontsize=7.2)
a2.set_title("Com hedge delta", fontsize=7.6)
pct(a2, "x")
for ax in (a1, a2):
    ax.set_xlabel("resultado / prêmio recebido")
    ax.set_yticks([])
salvar(fig, "cap05/hedge")
