"""Modelo de mercado: dispersão, reta de MQO e beta móvel."""
import sys; sys.path.insert(0, "..")
from estilo import *
import statsmodels.api as sm

p = precos("PETR4.SA", "^BVSP").dropna()
Rt = p.pct_change().dropna()
R = Rt.loc["2019":"2024"]
m = sm.OLS(R["PETR4.SA"], sm.add_constant(R["^BVSP"])).fit()
a, b = m.params

fig, (a1, a2) = figura(altura=55, ncols=2,
                       gridspec_kw={"width_ratios": [1, 1.15]})
a1.scatter(R["^BVSP"], R["PETR4.SA"], s=3, color=ROXO,
           alpha=0.35, linewidths=0)
x = np.array([-0.16, 0.15])
a1.plot(x, a + b * x, color=TINTA, lw=0.9)
rot = f"inclinação\n$\\hat\\beta$ = {b:.2f}".replace(".", ",")
a1.annotate(rot,
            (0.085, a + b * 0.085), xytext=(-2, -30),
            textcoords="offset points", fontsize=7, ha="left")
a1.set_xlabel("retorno diário do Ibovespa")
a1.set_ylabel("retorno diário de PETR4")
a1.set_title("2019 a 2024")
pct(a1, "x"); pct(a1, "y")
a1.grid(False)
a1.axhline(0, color=CINZA_CLARO, lw=0.5, zorder=0)
a1.axvline(0, color=CINZA_CLARO, lw=0.5, zorder=0)

beta = (Rt["PETR4.SA"].rolling(252).cov(Rt["^BVSP"])
        / Rt["^BVSP"].rolling(252).var()).dropna()
a2.plot(beta.index, beta, color=ROXO)
a2.axhline(1, color=CINZA, lw=0.7, ls=(0, (4, 3)))
a2.set_title("Beta móvel (252 pregões)")
virgula(a2)
a2.xaxis.set_major_locator(mpl.dates.YearLocator(3))
a2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
salvar(fig, "cap09/beta")
print(f"beta {b:.4f} alfa {a:.5f} R2 {m.rsquared:.3f}")
print(f"beta móvel: mín {beta.min():.2f} em "
      f"{beta.idxmin():%Y-%m}, máx {beta.max():.2f} em "
      f"{beta.idxmax():%Y-%m}")
