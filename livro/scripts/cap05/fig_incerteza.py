"""Erro de estimação de medidas de cauda: bootstrap com 250 dias e com a amostra inteira."""
import sys; sys.path.insert(0, "..")
from estilo import *
from carteira import perda

L = perda["2015":"2024"].values / 1000
rng = np.random.default_rng(11)

def medidas(x):
    v95, v99 = np.quantile(x, [0.95, 0.99])
    return x.std(ddof=1), v95, v99, x[x >= v99].mean()

nomes = ["desvio-padrão", "VaR 95%", "VaR 99%", "ES 99%"]
fig, ax = figura(altura=50)
for j, (n, cor) in enumerate(((250, ROXO), (len(L), OCRE))):
    b = np.array([medidas(rng.choice(L, n)) for _ in range(5000)])
    lo, med, hi = np.percentile(b, [5, 50, 95], axis=0)
    for i in range(4):
        y = 3 - i + (0.16 if j == 0 else -0.16)
        ax.plot([lo[i], hi[i]], [y, y], color=cor, lw=3, solid_capstyle="butt")
        ax.plot([med[i]], [y], "o", color="white", ms=3.2, mec=TINTA, mew=0.7, zorder=3)
    print(n, "p5", lo.round(1), "med", med.round(1), "p95", hi.round(1), "largura/mediana", ((hi - lo) / med).round(2))
ax.set_yticks(range(4)); ax.set_yticklabels(nomes[::-1])
ax.text(hi[3] + 2, 0 - 0.16, f"{len(L)} dias", fontsize=7, va="center")
ax.text(102, 0 + 0.16 + 0.27, "250 dias", fontsize=7, va="center", ha="right")
ax.set_ylim(-0.6, 3.5); ax.set_xlim(0, None)
ax.grid(False); ax.grid(axis="x")
ax.set_xlabel("estimativa (R$ mil): mediana e intervalo de 90% do bootstrap")
salvar(fig, "cap05/incerteza")
