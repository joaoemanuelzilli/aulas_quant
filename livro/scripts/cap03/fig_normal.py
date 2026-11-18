"""Padronizar: a mesma área em duas escalas."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy.stats import norm

mu, sigma = 0.10, 0.48
x0 = np.log(0.70)
z0 = (x0 - mu) / sigma
fig, (a1, a2) = figura(altura=50, ncols=2)
x = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 600)
a1.plot(x, norm.pdf(x, mu, sigma), color=ROXO)
a1.fill_between(x[x <= x0], norm.pdf(x[x <= x0], mu, sigma), color=LILAS)
a1.axvline(x0, color=CINZA, lw=0.6)
a1.set_title("Log-retorno de um ano, N(0,10; 0,48²)")
a1.set_xlabel("log-retorno r")
virgula(a1, "x", 1)
a1.text(x0 - 0.05, 0.62, f"r < ln 0,70\n= {x0:.3f}".replace(".", ",").replace("-", "−"),
        ha="right", fontsize=6.8)
z = np.linspace(-3.5, 3.5, 600)
a2.plot(z, norm.pdf(z), color=ROXO)
a2.fill_between(z[z <= z0], norm.pdf(z[z <= z0]), color=LILAS)
a2.axvline(z0, color=CINZA, lw=0.6)
a2.set_title("Padronizada, Z = (r − 0,10)/0,48")
a2.set_xlabel("z")
virgula(a2, "x", 0)
a2.text(z0 - 0.12, 0.30, f"z < {z0:.3f}".replace(".", ",").replace("-", "−"), ha="right", fontsize=6.8)
for a, xx, yy, tx, ty in ((a1, x0 - 0.3, 0.06, -1.25, 0.35), (a2, z0 - 0.6, 0.06, -2.6, 0.17)):
    a.annotate(f"área\n{norm.cdf(z0) * 100:.1f}%".replace(".", ","), (xx, yy), xytext=(tx, ty),
               textcoords="data", fontsize=6.8, ha="center",
               arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
    a.set_yticks([])
    a.grid(False)
    a.spines["left"].set_visible(False)
salvar(fig, "cap03/normal")
print(f"x0 {x0:.4f} z0 {z0:.4f} área {norm.cdf(z0):.4f}")
