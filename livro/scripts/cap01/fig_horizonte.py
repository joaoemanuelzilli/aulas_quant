"""Horizonte: a chance de perda cai, a dispersão do patrimônio cresce."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy.stats import norm

MU, SIGMA = 0.10, 0.20          # log-retorno esperado e volatilidade, ao ano
T = np.linspace(0, 30, 301)
media, desvio = MU * T, SIGMA * np.sqrt(T)

fig, (a1, a2) = figura(altura=50, ncols=2)
a1.fill_between(T, np.exp(media - desvio), np.exp(media + desvio), color=LILAS)
a1.plot(T, np.exp(media), color=ROXO)
a1.axhline(1, color=TINTA, lw=0.6)
a1.set_yscale("log")
a1.set_yticks([1, 2, 5, 10, 20, 50]); a1.set_yticklabels(["1", "2", "5", "10", "20", "50"]); a1.minorticks_off()
a1.set_title("R$ 1 aplicado (escala log)")
a1.annotate("mediana", (22, np.exp(MU * 22)), xytext=(-4, 10), textcoords="offset points", ha="right", fontsize=7.2)
a1.text(1.2, 55, "faixa lilás: um desvio-padrão\npara cada lado da mediana", fontsize=6.8, va="top")
a2.plot(T[1:], norm.cdf(-MU * T[1:] / desvio[1:]), color=ROXO)
a2.set_title("Probabilidade de perda")
a2.set_ylim(0, 0.5)
pct(a2)
for T0 in (1, 10):
    pr = norm.cdf(-MU * T0 / (SIGMA * np.sqrt(T0)))
    a2.plot([T0], [pr], "o", color=ROXO, ms=3.5)
    a2.annotate(f"{pr * 100:.0f}% em {T0} ano" + ("s" if T0 > 1 else ""), (T0, pr), xytext=(6, 4),
                textcoords="offset points", fontsize=6.8)
for a in (a1, a2):
    a.set_xlabel("horizonte (anos)"); a.set_xlim(0, 30); a.set_xticks([0, 10, 20, 30])
salvar(fig, "cap01/horizonte")
for T0 in (1, 10, 30):
    d = SIGMA * np.sqrt(T0)
    print(f"T={T0}: P(perda) {norm.cdf(-MU * T0 / d):.1%}; faixa {np.exp(MU * T0 - d):.2f} a {np.exp(MU * T0 + d):.2f}")
