"""Distribuição das perdas diárias da carteira, com VaR e ES históricos."""
import sys; sys.path.insert(0, "..")
from estilo import *
from carteira import perda

L = perda["2015":"2024"] / 1000
v95, v99 = np.quantile(L, [0.95, 0.99])
es95 = L[L >= v95].mean()
lim = 80
bins = np.arange(-lim, lim + 2.5, 2.5)
fig, ax = figura(altura=56)
cont, _ = np.histogram(L.clip(-lim, lim - 0.01), bins)
centros = bins[:-1] + 1.25
cores = [OCRE if c > v95 else ROXO for c in centros]
ax.bar(centros, cont, width=2.2, color=cores)
topo = cont.max()
for x, nome, y in ((v95, "VaR 95%", 0.60), (v99, "VaR 99%", 0.30)):
    ax.plot([x, x], [0, topo * y], color=TINTA, lw=0.8)
    ax.text(x + 1.2, topo * y, f"{nome}\nR\\$ {x:.1f} mil".replace(".", ","), fontsize=7, va="top")
ax.text(-lim, topo * 0.9, "ganhos", fontsize=7.2, color=CINZA)
ax.text(lim, topo * 0.9, "perdas", fontsize=7.2, color=CINZA, ha="right")
n_fora = int((L > lim).sum())
ax.annotate(f"{n_fora} dias de março de 2020\ncom perda acima de R\\$ {lim} mil", (lim - 1.25, cont[-1]), xytext=(lim, topo * 0.80),
            fontsize=6.8, ha="right", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
n_esq = int((L < -lim).sum())
ax.annotate(f"{n_esq} dia com ganho\nacima de R\\$ {lim} mil", (-lim + 1.25, cont[0]), xytext=(-lim, topo * 0.45),
            fontsize=6.8, ha="left", va="top", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.set_xlabel("perda diária da carteira de R\\$ 1 milhão (R\\$ mil)")
ax.set_ylabel("número de dias")
ax.set_xlim(-lim - 2, lim + 2)
salvar(fig, "cap05/perdas")
print(v95, v99, es95, n_fora, (L > v95).sum(), (L < -lim).sum(), L[L > lim])
