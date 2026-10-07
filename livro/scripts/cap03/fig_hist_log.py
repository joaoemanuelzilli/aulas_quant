"""Histograma do Ibovespa contra a normal, em escala linear e log."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy import stats

r = np.log(precos("^BVSP")["^BVSP"].dropna()).diff().dropna()
mu, s = r.mean(), r.std()
bordas = np.arange(-0.17, 0.1401, 0.005)
x = np.linspace(-0.17, 0.14, 800)
dens = stats.norm.pdf(x, mu, s)

fig, (a1, a2) = figura(altura=56, ncols=2)
for ax in (a1, a2):
    ax.hist(r, bins=bordas, density=True, color=ROXO, alpha=0.55)
    ax.plot(x, dens, color=TINTA, lw=0.9)
    pct(ax, "x")
    ax.set_xlabel("log-retorno diário")
a1.set_xlim(-0.08, 0.08)
a1.set_title("Escala linear")
a1.set_ylabel("densidade")
a1.annotate("normal com a\nmesma média\ne desvio", (0.021, stats.norm.pdf(0.021, mu, s)),
            xytext=(0.034, 21), fontsize=6.8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a2.set_yscale("log")
a2.set_ylim(0.03, 60)
a2.set_xlim(-0.175, 0.145)
a2.set_title("Escala logarítmica")
a2.set_yticks([0.1, 1, 10]); a2.set_yticklabels(["0,1", "1", "10"])
a2.minorticks_off()
a2.annotate("a normal não\nprevê nada aqui", (-0.105, 0.09), xytext=(-0.165, 3.5),
            fontsize=6.8, arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
salvar(fig, "cap03/hist_log")
h, _ = np.histogram(r, bins=bordas)
print("menor densidade não nula do histograma:", 1 / (len(r) * 0.005))
print("retornos fora de ±8%:", (r.abs() > 0.08).sum())
print("normal: |r| em que a densidade cai a 0,03:",
      s * np.sqrt(-2 * np.log(0.03 * s * np.sqrt(2 * np.pi))))
