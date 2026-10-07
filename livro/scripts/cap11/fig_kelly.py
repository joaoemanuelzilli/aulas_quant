"""Taxa de crescimento em função da fração apostada (aposta binária)."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l05_kelly import crescimento, f_kelly

f = np.linspace(0, 0.26, 300)
fig, ax = figura(altura=54)
ax.axhline(0, color=TINTA, lw=0.6)
ax.plot(f, crescimento(f) * 100, color=ROXO)
for mult, nome, dy, ha in ((0.5, "meio Kelly", 9, "right"), (1.0, "Kelly", 9, "center"), (2.0, "dobro de Kelly", 9, "left")):
    x = mult * f_kelly; y = crescimento(x) * 100
    ax.plot([x], [y], "o", color=OCRE, ms=4, zorder=5)
    ax.annotate(nome, (x, y), xytext=(-5 if ha == "right" else (5 if ha == "left" else 0), dy if ha == "center" else 3), textcoords="offset points", ha=ha, fontsize=7)
ax.axvspan(2 * f_kelly, 0.26, color=CINZA_CLARO, alpha=0.45, zorder=0)
ax.text(0.23, 0.38, "a riqueza encolhe\nno longo prazo", fontsize=6.8, ha="center", va="center")
ax.set_xlabel("fração do capital apostada a cada rodada, $f$"); ax.set_ylabel("crescimento esperado do log\nda riqueza por aposta (%)")
pct(ax, "x"); virgula(ax, "y", 1); ax.set_xlim(0, 0.26); ax.set_ylim(-0.75, 0.7)
salvar(fig, "cap11/kelly")
