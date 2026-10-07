"""Regressão espúria: estatística t em níveis e em diferenças."""
import sys; sys.path.insert(0, "..")
from estilo import *
import statsmodels.api as sm

rng = np.random.default_rng(42)          # mesma semente do l01_espuria.py
T, N = 500, 2000
tn, td, r2, pares = [], [], [], []
for i in range(N):
    x = rng.standard_normal(T).cumsum(); y = rng.standard_normal(T).cumsum()
    m = sm.OLS(y, sm.add_constant(x)).fit()
    tn.append(m.tvalues[1]); r2.append(m.rsquared)
    td.append(sm.OLS(np.diff(y), sm.add_constant(np.diff(x))).fit().tvalues[1])
    if i < 40: pares.append((x, y, m.rsquared, m.tvalues[1]))
tn, td = np.array(tn), np.array(td)
x, y, R2, tt = max(pares, key=lambda p: p[2])   # o par mais "convincente" entre os 40 primeiros
print(f"par mostrado: R2 = {R2:.2f}, t = {tt:.1f}, corr. das diferenças = {np.corrcoef(np.diff(x), np.diff(y))[0, 1]:.3f}")
print(f"rejeições: níveis {np.mean(abs(tn) > 1.96):.1%}  diferenças {np.mean(abs(td) > 1.96):.1%}")

fig, (a, b) = figura(altura=56, ncols=2, gridspec_kw={"width_ratios": [1, 1.15]})
a.plot(x, color=ROXO, lw=0.9); a.plot(y, color=OCRE, lw=0.9)
a.set_title("Dois passeios independentes")
a.set_xlabel("tempo $t$")
a.text(0.04, 0.93, f"$R^2$ = {R2:.2f}".replace(".", ","), transform=a.transAxes, fontsize=7.2, va="top")
bins = np.linspace(-60, 60, 61)
print(f"|t| em níveis acima de 60: {np.mean(abs(tn) > 60):.2%}; máximo {abs(tn).max():.0f}")
b.hist(tn, bins=bins, color=ROXO, alpha=0.85)
b.hist(td, bins=bins, color=CINZA, alpha=0.9)
b.set_title("Estatística $t$ em 2\u2009000 simulações")
b.set_xlabel("$t$ do coeficiente angular"); b.set_yticks([])
b.grid(False)
b.annotate("em diferenças", (1.5, 0.93), xycoords=("data", "axes fraction"), xytext=(18, 0), textcoords="offset points",
           fontsize=7, va="center", arrowprops=dict(arrowstyle="-", color=CINZA, lw=0.5))
b.text(0.03, 0.30, "em níveis", transform=b.transAxes, fontsize=7)
salvar(fig, "cap04/espuria")
