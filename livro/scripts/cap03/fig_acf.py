"""Autocorrelação dos retornos e dos retornos ao quadrado (Ibovespa)."""
import sys; sys.path.insert(0, "..")
from estilo import *
from statsmodels.tsa.stattools import acf

r = np.log(precos("^BVSP")["^BVSP"].dropna()).diff().dropna()
L = 40
faixa = 1.96 / np.sqrt(len(r))
fig, (a1, a2) = figura(altura=58, ncols=2, sharey=True)
for ax, x, tit in [(a1, r, "Retornos"), (a2, r ** 2, "Retornos ao quadrado")]:
    rho = acf(x, nlags=L)[1:]
    ax.bar(np.arange(1, L + 1), rho, width=0.62, color=ROXO)
    ax.axhspan(-faixa, faixa, color=CINZA_CLARO, alpha=0.6, zorder=0)
    ax.axhline(0, color=TINTA, lw=0.6)
    ax.set_title(tit)
    ax.set_xlabel("defasagem (dias)")
    ax.set_xlim(0, L + 1)
    print(tit, rho[:5].round(3), "lag 20:", rho[19].round(3), "lag 40:", rho[39].round(3),
          "fora da faixa:", (np.abs(rho) > faixa).sum(), "máx |rho|:", np.abs(rho).max().round(3))
a1.set_ylabel("autocorrelação")
a1.set_ylim(-0.12, 0.56)
virgula(a1)
a1.text(22, 0.075, "faixa de 95% sob\nindependência", fontsize=6.8, color=CINZA, va="bottom")
salvar(fig, "cap03/acf")
