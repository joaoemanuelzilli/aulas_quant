"""Quantis da normal e da t de mesma variância, contra os quantis empíricos."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy import stats
from carteira import perda

L = perda["2015":"2024"]
nu = stats.t.fit(L)[0]
cauda = np.logspace(np.log10(0.10), np.log10(0.001), 200)      # 1 - alfa
qn = stats.norm.ppf(1 - cauda)
qt = stats.t.ppf(1 - cauda, nu) * np.sqrt((nu - 2) / nu)
niveis = np.array([0.10, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001])
qe = (np.quantile(L, 1 - niveis) - L.mean()) / L.std()

fig, ax = figura(altura=56)
ax.plot(cauda, qn, color=CINZA, ls=(0, (4, 3)), lw=1)
ax.plot(cauda, qt, color=ROXO)
ax.plot(niveis, qe, "o", color=OCRE, ms=3.6, zorder=3)
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xticks(niveis); ax.set_xticklabels(["90%", "95%", "97,5%", "99%", "99,5%", "99,8%", "99,9%"]); ax.minorticks_off()
ax.set_xlim(0.115, 0.00045)
rotulo_final(ax, cauda[-1], qn[-1], "normal", CINZA)
rotulo_final(ax, cauda[-1], qt[-1], f"t, $\\nu$ = {nu:.1f}".replace(".", ","), ROXO)
ax.annotate("carteira\n(quantis empíricos)", (niveis[-2], qe[-2]), xytext=(-10, 16), textcoords="offset points",
            fontsize=7, ha="right", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
cruz = cauda[np.argmin(np.abs(qn - qt))]
ax.set_xlabel("nível de confiança $\\alpha$")
ax.set_ylabel("quantil da perda (em desvios-padrão)")
virgula(ax, "y", 0)
salvar(fig, "cap05/normal_t")
print("nu", nu, "cruzamento em alfa =", 1 - cruz, dict(zip(niveis, qe.round(2))))
