"""Ruído branco (estacionário) contra passeios aleatórios (não estacionários)."""
import sys; sys.path.insert(0, "..")
from estilo import *

rng = np.random.default_rng(4)
T = 500
eps = rng.standard_normal((5, T))
t = np.arange(1, T + 1)
fig, (a, b) = figura(altura=75, nrows=2, sharex=True,
                     gridspec_kw={"height_ratios": [1, 1.7]})
a.plot(t, eps[0], color=ROXO, lw=0.6)
a.axhline(0, color=CINZA, lw=0.6)
a.set_ylim(-4.2, 4.2); a.set_yticks([-2, 0, 2])
a.set_title("Ruído branco: média e dispersão não mudam")
b.fill_between(t, -2 * np.sqrt(t), 2 * np.sqrt(t), color=LILAS, zorder=0)
for i, c in enumerate(CATEGORICAS):
    b.plot(t, eps[i].cumsum(), color=c, lw=0.9)
b.axhline(0, color=CINZA, lw=0.6)
b.set_title("Cinco passeios aleatórios: a soma acumulada do ruído")
b.text(330, 2 * np.sqrt(330) + 2.5, r"faixa $\pm 2\sqrt{t}$", fontsize=7, color=CINZA, rotation=9)
b.set_xlabel("tempo $t$"); b.set_xlim(0, T)
b.set_ylim(-52, 52)
salvar(fig, "cap04/rb_pa")
print("valores finais dos passeios:", eps.cumsum(axis=1)[:, -1].round(1))
