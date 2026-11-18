"""Dias de alta de PETR4 em blocos de 21 pregões contra a binomial."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy.stats import binom

petr = precos("PETR4.SA")["PETR4.SA"].dropna()
alta = (petr.pct_change().dropna() > 0).to_numpy()
n = 21
blocos = len(alta) // n
K = alta[:blocos * n].reshape(blocos, n).sum(axis=1)
p = alta.mean()
k = np.arange(0, n + 1)
obs = np.array([np.mean(K == i) for i in k])

fig, ax = figura(altura=55)
ax.bar(k, obs, width=0.7, color=LILAS, edgecolor=ROXO, linewidth=0.6)
ax.plot(k, binom.pmf(k, n, p), "o-", color=TINTA, ms=3, lw=0.8)
ax.annotate(f"{blocos} blocos de 2010 a 2025", (7, obs[7]), xytext=(4.5, 0.13), textcoords="data",
            ha="center", fontsize=6.8, arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.annotate(f"binomial com n = 21 e p = {p:.3f}".replace(".", ","), (13, binom.pmf(13, n, p)),
            xytext=(16, 14), textcoords="offset points", fontsize=6.8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.set_xlim(2.5, 19.5)
ax.set_xticks(range(3, 20, 2))
ax.set_xlabel("dias de alta em 21 pregões")
ax.set_ylabel("fração dos blocos")
pct(ax)
ax.set_yticks([0, 0.05, 0.10, 0.15])
salvar(fig, "cap03/binomial")
print("observado:", dict(zip(k[obs > 0], (obs[obs > 0] * blocos).astype(int))))
print("moda observada", k[obs.argmax()], "binomial", k[binom.pmf(k, n, p).argmax()])
