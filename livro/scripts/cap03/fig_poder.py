"""Erros tipo I e II e o poder do teste t para a média do retorno."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy import stats

zc = stats.norm.ppf(0.975)
d = 2.0                                  # deslocamento sob H1 (ilustração)
x = np.linspace(-4.2, 6.2, 800)
fig, (a1, a2) = figura(altura=62, ncols=2, gridspec_kw={"width_ratios": [1.15, 1]})
f0, f1 = stats.norm.pdf(x), stats.norm.pdf(x, d)
a1.plot(x, f0, color=CINZA, lw=1.1)
a1.plot(x, f1, color=ROXO)
a1.fill_between(x, f0, where=np.abs(x) >= zc, color=CINZA, alpha=0.55)
a1.fill_between(x, f1, where=np.abs(x) <= zc, color=ROXO, alpha=0.22)
a1.fill_between(x, f1, where=x >= zc, color=ROXO, alpha=0.75)
for v in (-zc, zc):
    a1.axvline(v, color=TINTA, lw=0.6, ls=(0, (3, 2)))
a1.set_ylim(0, 0.56)
a1.set_xlim(-4.2, 6.2)
a1.set_yticks([])
a1.grid(False)
a1.spines["left"].set_visible(False)
a1.set_xlabel("estatística $t$")
a1.set_title("Duas hipóteses, uma regra de decisão")
a1.text(-2.25, 0.34, "sob $H_0$", ha="right", fontsize=7.2, color=CINZA)
a1.text(3.3, 0.37, "sob $H_1$", ha="left", fontsize=7.2)
a1.annotate("erro tipo I\n($\\alpha/2$ por\ncauda)", (-2.25, 0.012), xytext=(-4.1, 0.12), fontsize=6.6,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a1.annotate("erro tipo II ($\\beta$)", (1.0, 0.12), xytext=(0.2, 0.49), fontsize=6.6, ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a1.annotate("poder\n($1-\\beta$)", (3.0, 0.1), xytext=(4.6, 0.24), fontsize=6.6,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))

anos = np.linspace(0.5, 40, 400)
for sr, cor in [(1.0, ROXO), (0.5, OCRE), (0.25, VERDE)]:
    nc = sr * np.sqrt(anos)              # valor esperado de t
    poder = stats.norm.sf(zc - nc) + stats.norm.cdf(-zc - nc)
    a2.plot(anos, poder, color=cor)
    rotulo_final(a2, 40, poder[-1] - (0.035 if sr == 0.5 else 0), f"{sr}".replace(".", ","), cor)
    alvo = ((zc + stats.norm.ppf(0.8)) / sr) ** 2
    p10 = stats.norm.sf(zc - sr * np.sqrt(10)) + stats.norm.cdf(-zc - sr * np.sqrt(10))
    print(f"razão {sr}: poder de 80% com {alvo:.1f} anos; poder com 10 anos = {p10:.3f}")
a2.axhline(0.8, color=CINZA, lw=0.6, ls=(0, (3, 2)))
a2.set_xlim(0, 47); a2.set_ylim(0, 1.02)
a2.spines["bottom"].set_bounds(0, 40)
a2.set_xticks([0, 10, 20, 30, 40])
pct(a2)
a2.set_xlabel("anos de dados diários")
a2.set_title("Poder, por razão $\\mu/\\sigma$ anual")
salvar(fig, "cap03/poder")
print("beta do painel esquerdo:", stats.norm.cdf(zc - d) - stats.norm.cdf(-zc - d))
