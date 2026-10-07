"""O que Pearson, Spearman e Kendall enxergam em quatro nuvens."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy import stats

rng = np.random.default_rng(7)
n = 400
x = rng.normal(size=n)
casos = {}
casos["Linear"] = (x, 0.7 * x + np.sqrt(1 - 0.49) * rng.normal(size=n))
casos["Monótona, não linear"] = (x, np.exp(1.5 * x))
xu = np.linspace(-2.5, 2.5, n)             # simétrico em torno de zero
casos["Dependência em U"] = (xu, xu ** 2 + 0.4 * rng.normal(size=n))
xo, yo = rng.normal(size=150), rng.normal(size=150)
xo[0], yo[0] = 10.0, 10.0                 # um único ponto extremo
casos["Um ponto extremo"] = (xo, yo)

fig, eixos = figura(altura=54, largura="larga", ncols=4)
for ax, (tit, (a, b)) in zip(eixos, casos.items()):
    ax.plot(a, b, "o", ms=2.0, color=ROXO, alpha=0.55, mew=0)
    ax.set_title(tit)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    pe, sp, ke = stats.pearsonr(a, b)[0], stats.spearmanr(a, b)[0], stats.kendalltau(a, b)[0]
    txt = (f"Pearson {pe:.2f}\nSpearman {sp:.2f}\nKendall {ke:.2f}").replace(".", ",").replace("-", "−")
    ax.set_xlabel(txt, color=TINTA, fontsize=7.2, loc="left", linespacing=1.35)
    print(f"{tit:24s} Pearson {pe:.3f}  Spearman {sp:.3f}  Kendall {ke:.3f}")
salvar(fig, "cap03/pearson")
print("sem o ponto extremo:", stats.pearsonr(xo[1:], yo[1:])[0])
