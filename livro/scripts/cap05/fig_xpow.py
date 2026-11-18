"""A função x^(1/x) e seu máximo em x = e."""
import sys; sys.path.insert(0, "..")
from estilo import *

x = np.linspace(0.4, 8, 400)
fig, ax = figura(altura=36, largura="margem")
ax.plot(x, x ** (1 / x), color=ROXO, lw=1.2)
for v, rot in [(np.e, "$e$"), (np.pi, "$\\pi$")]:
    ax.plot(v, v ** (1 / v), "o", color=TINTA, ms=2.6)
    ax.annotate(rot, (v, v ** (1 / v)),
                xytext=(-6 if v < 3 else 3, 3),
                textcoords="offset points", fontsize=7.2)
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_ylim(0.0, 1.75)
virgula(ax, "y", 1)
ax.set_xlabel("$x$")
salvar(fig, "cap05/x_pow")
