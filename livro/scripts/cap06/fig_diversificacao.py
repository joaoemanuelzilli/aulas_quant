"""Volatilidade de carteiras igualmente ponderadas contra N."""
import sys; sys.path.insert(0, "..")
from itertools import combinations

from estilo import *
from carteira import retornos

r = retornos("2015", "2024")
cov = r.cov().values * 252
n = len(cov)
fora = ~np.eye(n, dtype=bool)
var_media, cov_media = np.diag(cov).mean(), cov[fora].mean()

fig, ax = figura(altura=62)
for k in range(1, n + 1):
    vols = [np.sqrt(cov[np.ix_(s, s)].mean())
            for s in combinations(range(n), k)]
    ax.plot([k] * len(vols), vols, "o", color=CINZA_CLARO, ms=2.6,
            zorder=1)
    print(k, len(vols), f"média das vols {np.mean(vols):.4f}",
          f"mín {min(vols):.4f} máx {max(vols):.4f}")
N = np.arange(1, 31)
curva = np.sqrt(var_media / N + (1 - 1 / N) * cov_media)
ax.plot(N, curva, color=ROXO, zorder=3)
ax.plot(N[:n], curva[:n], "o", color=ROXO, ms=3.4, zorder=4)
piso = np.sqrt(cov_media)
ax.axhline(piso, color=TINTA, lw=0.7, ls=(0, (4, 3)))
ax.text(30, piso + 0.03, "piso: raiz da covariância média = "
        + f"{piso:.1%}".replace(".", ","), ha="right", va="bottom",
        fontsize=7)
ax.fill_between(N, piso, curva, color=LILAS, zorder=0)
ax.annotate("risco diversificável", (4.5, 0.262),
            xytext=(9, 0.40), fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.text(15, 0.17, "risco sistemático: não cai com $N$",
        fontsize=7, ha="center")
ax.text(9, 0.60, "cada ponto cinza é uma das 127 carteiras\n1/N "
        "possíveis com as sete ações", fontsize=6.8, color=CINZA,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.set_xlim(0, 30.6); ax.set_ylim(0.10, 0.70)
ax.set_xlabel("número de ações na carteira ($N$)")
ax.set_ylabel("volatilidade anual")
pct(ax)
salvar(fig, "cap06/diversificacao")
print(f"variância média {var_media:.4f} (vol {np.sqrt(var_media):.4f})")
print(f"covariância média {cov_media:.4f} (piso {piso:.4f})")
print("curva N=1,7,30:", curva[[0, 6, 29]].round(4))
p = precos("BOVA11.SA", inicio="2015", fim="2024").dropna()
print("vol BOVA11:", float(p.pct_change().std().iloc[0] * np.sqrt(252)))
