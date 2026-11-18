"""Teorema central do limite: a soma de n dados, distribuição exata."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy.stats import norm

um = np.full(6, 1 / 6)                 # P(X = 1), ..., P(X = 6)
fig, eixos = figura(altura=42, largura="larga", ncols=4)
for ax, n in zip(eixos, (1, 2, 5, 30)):
    pmf = np.array([1.0])
    for _ in range(n):
        pmf = np.convolve(pmf, um)      # soma de mais um dado
    s = np.arange(n, 6 * n + 1)        # valores possíveis da soma
    mu, sd = 3.5 * n, np.sqrt(35 / 12 * n)
    ax.bar(s, pmf, width=0.8 if n < 30 else 1.0, color=LILAS, edgecolor=ROXO, linewidth=0.4 if n < 30 else 0)
    x = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
    ax.plot(x, norm.pdf(x, mu, sd), color=TINTA, lw=0.8)
    ax.set_title(f"n = {n}")
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(mu - 3.6 * sd, mu + 3.6 * sd) if n > 1 else ax.set_xlim(0.2, 6.8)
    print(n, "soma de", s[0], "a", s[-1], "média", mu, "dp", round(sd, 3), "max pmf", pmf.max().round(4))
eixos[0].set_xticks(range(1, 7))
eixos[3].set_xticks([80, 105, 130])
eixos[0].set_xlabel("um dado")
for ax in eixos[1:]:
    ax.set_xlabel("soma dos n dados")
salvar(fig, "cap03/tcl")
