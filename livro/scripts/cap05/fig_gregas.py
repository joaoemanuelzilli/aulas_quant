"""Delta e gama da opção de compra para três prazos."""
import sys; sys.path.insert(0, "..")
from estilo import *
from calculo import gregas

K, r, sigma = 30.0, 0.14, 0.30
S = np.linspace(20, 40, 401)
fig, (a1, a2) = figura(altura=58, ncols=2)
for du, cor, rot in [(63, ROXO, "63 du"), (21, OCRE, "21 du"),
                     (5, VERDE, "5 du")]:
    g = gregas(S, K, du / 252, r, sigma)
    a1.plot(S, g["delta"], color=cor, lw=1.2)
    a2.plot(S, g["gama"], color=cor, lw=1.2)
    k = np.argmax(g["gama"])
    a2.annotate(rot, (S[k], g["gama"][k]),
                xytext=(-6, 4) if du == 63 else (6, 3),
                ha="right" if du == 63 else "left",
                textcoords="offset points", fontsize=7.2, va="center")
a1.annotate("5 du", (30.9, 0.8), xytext=(33.5, 0.55), fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a1.annotate("63 du", (25, 0.2), xytext=(20.5, 0.38), fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a1.set_title("Delta, $\\partial C/\\partial S$", fontsize=7.6)
a2.set_title("Gama, $\\partial^2 C/\\partial S^2$", fontsize=7.6)
for ax in (a1, a2):
    ax.set_xlabel("preço da ação $S$ (R\\$)")
    ax.axvline(K, color=CINZA_CLARO, lw=0.6, zorder=0)
    virgula(ax, "x", 0)
virgula(a1, "y", 1); virgula(a2, "y", 2)
salvar(fig, "cap05/gregas")
