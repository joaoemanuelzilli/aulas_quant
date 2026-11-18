"""Bayes: a probabilidade a posteriori depende da taxa-base."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l02_bayes import posterior, base

prior = np.logspace(-3, np.log10(0.5), 400)
fig, ax = figura(altura=58, constrained_layout=False)
fig.subplots_adjust(left=0.11, right=0.76, bottom=0.17, top=0.96)
ax.plot(prior, prior, color=CINZA, lw=0.8, ls="--")
rotulo_final(ax, prior[-1], prior[-1], "alarme inútil", CINZA)
casos = [(0.80, 0.10, ROXO, "alarme falso em 10%"),
         (0.80, 0.02, OCRE, "alarme falso em 2%")]
for sens, falso, cor, rot in casos:
    y = posterior(prior, sens, falso)
    ax.plot(prior, y, color=cor)
    rotulo_final(ax, prior[-1], y[-1], rot, cor)
pb = posterior(base, 0.80, 0.10)
ax.plot([base], [pb], "o", color=ROXO, ms=3.5)
ax.annotate(f"taxa-base do Ibovespa, {base * 100:.1f}%:\n"
            f"P(queda | alarme) = {pb * 100:.1f}%".replace(".", ","),
            (base, pb), xytext=(0.0011, 0.66), textcoords="data", ha="left", fontsize=6.8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.set_xscale("log")
ax.set_xlim(1e-3, 0.5)
ax.set_ylim(0, 1)
ax.set_xticks([0.001, 0.01, 0.1, 0.5])
ax.set_xticklabels(["0,1%", "1%", "10%", "50%"])
ax.minorticks_off()
pct(ax)
ax.set_xlabel("taxa-base: P(queda) antes de ver o alarme (escala log)")
ax.set_ylabel("P(queda | alarme)")
salvar(fig, "cap03/bayes")
print(f"base {base:.4f}  posterior {pb:.4f}; com falso 2%: {posterior(base, 0.8, 0.02):.4f}")
