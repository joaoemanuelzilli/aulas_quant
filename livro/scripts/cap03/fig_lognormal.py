"""O preço daqui a um ano sob log-retorno normal: lognormal."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy.stats import lognorm

mu, sigma, P0 = 0.10, 0.48, 30.0
dist = lognorm(s=sigma, scale=P0 * np.exp(mu))   # P_T = P0 exp(r)
mediana, media = dist.median(), dist.mean()
x = np.linspace(0.01, 120, 1200)
fig, ax = figura(altura=55)
ax.plot(x, dist.pdf(x), color=ROXO)
ax.fill_between(x[x <= 0.7 * P0], dist.pdf(x[x <= 0.7 * P0]), color=LILAS)
ymax = dist.pdf(x).max()
for v, nome, dx, dy, ha in ((P0, "hoje", -1.0, 1.10, "right"), (mediana, "mediana", 0, 1.26, "center"),
                           (media, "média", 1.0, 1.10, "left")):
    ax.plot([v, v], [0, ymax * (dy - 0.02)], color=CINZA, lw=0.6, ls="--" if nome != "hoje" else "-")
    ax.text(v + dx, ymax * dy, f"{nome}\nR$ {v:.2f}".replace(".", ","), ha=ha, fontsize=6.8, va="bottom")
ax.annotate(f"queda de mais de 30%:\n{dist.cdf(0.7 * P0) * 100:.1f}% de probabilidade".replace(".", ","),
            (17, dist.pdf(17) * 0.5), xytext=(-6, 30), textcoords="offset points", ha="right",
            fontsize=6.8, arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.annotate("cauda longa\npara cima", (75, dist.pdf(75)), xytext=(10, 20), textcoords="offset points",
            fontsize=6.8, arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.set_ylim(0, ymax * 1.52)
ax.set_xlim(0, 120)
ax.set_yticks([])
ax.grid(False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("preço daqui a um ano (R$)")
salvar(fig, "cap03/lognormal")
print(f"mediana {mediana:.2f} média {media:.2f} P(<21) {dist.cdf(21):.4f} moda {P0*np.exp(mu - sigma**2):.2f}")
