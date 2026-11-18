"""Quedas extremas do Ibovespa por mês: Poisson contra os dados."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy.stats import poisson

ibov = precos("^BVSP")["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()
N = (r < -0.03).groupby(r.index.to_period("M")).sum()
lam = N.mean()
k = np.arange(0, 10)
obs = np.array([np.sum(N == i) for i in k])
esp = len(N) * poisson.pmf(k, lam)

fig, (a1, a2) = figura(altura=56, ncols=2, gridspec_kw={"width_ratios": [1, 1.25]})
a1.bar(k[obs > 0], obs[obs > 0], width=0.7, color=LILAS, edgecolor=ROXO, linewidth=0.6)
a1.plot(k, esp, "o-", color=TINTA, ms=3, lw=0.8)
a1.set_yscale("log")
a1.set_ylim(1e-4, 400)
a1.set_yticks([1e-4, 1e-2, 1, 100])
a1.set_yticklabels(["0,0001", "0,01", "1", "100"])
a1.minorticks_off()
a1.set_xticks(k)
a1.set_xlabel("quedas extremas no mês")
a1.set_title("Número de meses (escala log)")
a1.annotate("Poisson com a\nmesma média", (6, esp[6]), xytext=(7.0, 0.004), textcoords="data",
            fontsize=6.8, arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a1.annotate("março\nde 2020", (9, 1.3), xytext=(7.3, 25), textcoords="data", ha="center",
            fontsize=6.8, arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
t = N.index.to_timestamp()
a2.vlines(t, 0, N.values, color=ROXO, lw=1.0)
a2.set_title("Quedas extremas em cada mês")
a2.set_ylim(0, 9.8)
a2.set_yticks([0, 3, 6, 9])
a2.xaxis.set_major_locator(mpl.dates.YearLocator(5))
a2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
salvar(fig, "cap03/poisson")
print("lambda", lam, "obs", obs, "esp", esp.round(4))
print("P(N>=9) Poisson:", poisson.sf(8, lam), " meses esperados:", len(N) * poisson.sf(8, lam))
