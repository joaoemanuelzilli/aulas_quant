"""Payoff no vencimento e preço de Black-Scholes antes dele."""
import sys; sys.path.insert(0, "..")
from estilo import *
from calculo import bs

K, r, sigma = 30.0, 0.14, 0.30
S = np.linspace(18, 42, 241)
fig, (a1, a2) = figura(altura=58, ncols=2, sharey=True)
for ax, tipo, pay, tit in [
        (a1, "compra", np.maximum(S - K, 0), "Opção de compra"),
        (a2, "venda", np.maximum(K - S, 0), "Opção de venda")]:
    ax.plot(S, pay, color=TINTA, lw=1.1)
    ax.plot(S, bs(S, K, 63 / 252, r, sigma, tipo), color=ROXO,
            lw=1.3)
    ax.set_title(tit, fontsize=7.6)
    ax.set_xlabel("preço da ação $S$ (R\\$)")
    ax.axvline(K, color=CINZA_CLARO, lw=0.6, zorder=0)
    virgula(ax, "x", 0)
a1.annotate("no vencimento", (38, 8), xytext=(35.5, 2.2),
            fontsize=7.2, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a1.annotate("3 meses antes", (26, bs(26, K, 0.25, r, sigma)),
            xytext=(18.5, 4.5), fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a1.annotate("$K = 30$", (K, 12), xytext=(3, 0),
            textcoords="offset points", fontsize=7.2, color=CINZA)
a2.annotate("3 meses antes: abaixo\ndo payoff se $S$ é baixo",
            (21, bs(21, K, 0.25, r, sigma, "venda")),
            xytext=(27.5, 8.5), fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a1.set_ylabel("valor (R\\$)")
virgula(a1, "y", 0)
salvar(fig, "cap05/payoff")
