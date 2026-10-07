"""Duzentas estratégias de puro ruído e suas estatísticas t."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy import stats

rng = np.random.default_rng(42)          # mesma semente de l07_multiplos.py
m, n = 200, 1260
ruido = rng.normal(0.0, 0.01, size=(n, m))
t = ruido.mean(axis=0) / (ruido.std(axis=0, ddof=1) / np.sqrt(n))
sig = np.abs(t) > stats.t.ppf(0.975, n - 1)
bonf = stats.norm.isf(0.05 / m / 2)
fig, ax = figura(altura=56)
i = np.arange(1, m + 1)
ax.fill_between([0, 201], -1.96, 1.96, color=CINZA_CLARO, alpha=0.45, zorder=0)
ax.plot(i[~sig], t[~sig], "o", ms=2.6, color=CINZA, mew=0)
ax.plot(i[sig], t[sig], "o", ms=3.6, color=ROXO, mew=0)
for v in (-bonf, bonf):
    ax.plot([0, 201], [v, v], color=TINTA, lw=0.6, ls=(0, (3, 2)))
ax.text(204, bonf, "limite de\nBonferroni", fontsize=6.8, va="center")
ax.text(204, 0, "$|t| < 1,96$", fontsize=6.8, va="center", color=CINZA)
ax.set_xlim(0, 236); ax.set_ylim(-4.3, 4.3)
ax.spines["bottom"].set_bounds(0, 200)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xlabel("estratégia (todas com retorno esperado igual a zero)")
ax.set_ylabel("estatística $t$")
ax.grid(False)
salvar(fig, "cap03/multiplos")
print("significativas a 5%:", sig.sum(), " positivas:", (t[sig] > 0).sum(), " Bonferroni:", bonf)
