"""Melhor Sharpe entre N estratégias sem habilidade."""
import sys; sys.path.insert(0, "..")
from estilo import *
from sobreajuste import maximo_esperado, ANOS

sim = pd.read_csv(DADOS / "c07_sobreajuste.csv")
ns = [1, 10, 100, 1000]
fig, ax = figura(altura=60)
larg = 0.34
for i, n in enumerate(ns):
    for col, cor, lado in [(f"dentro_{n}", ROXO, -1), (f"fora_{n}", CINZA, 1)]:
        v = sim[col]
        q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        x = i + lado * larg / 2 * 1.15
        ax.plot([x, x], [q05, q95], color=cor, lw=0.8)
        ax.add_patch(mpl.patches.Rectangle((x - larg / 2, q25), larg, q75 - q25,
                                           fc=cor, alpha=0.85 if cor == ROXO else 0.45, lw=0))
        ax.plot([x - larg / 2, x + larg / 2], [q50, q50], color="white", lw=1)
    teoria = maximo_esperado(n, 1 / np.sqrt(ANOS))
    ax.plot([i - larg * 1.15, i], [teoria, teoria], color=TINTA, lw=1.1, zorder=6)
ax.plot([-0.6, 3.45], [0, 0], color=TINTA, lw=0.6)
ax.set_xticks(range(len(ns))); ax.set_xticklabels(["1", "10", "100", "1.000"])
ax.set_xlabel("número de estratégias testadas ($N$)")
ax.set_ylabel("Sharpe anualizado")
ax.set_xlim(-0.6, 4.35); ax.set_ylim(-1.3, 2.3)
ax.spines["bottom"].set_bounds(-0.6, 3.5)
ax.text(3.5, 1.72, "a melhor,\ndentro da\namostra", fontsize=7, va="center")
ax.text(3.5, 0.0, "a mesma,\nnos 5 anos\nseguintes", fontsize=7, va="center")
ax.annotate("valor esperado teórico\ndo máximo", (1.82, maximo_esperado(100, 1 / np.sqrt(ANOS))),
            xytext=(-62, 26), textcoords="offset points", fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
virgula(ax)
salvar(fig, "cap07/sobreajuste")
for n in ns:
    print(n, sim[f"dentro_{n}"].mean().round(3), sim[f"fora_{n}"].mean().round(3),
          sim[f"dentro_{n}"].quantile([.05, .95]).round(2).tolist(), sim[f"fora_{n}"].std().round(3),
          (sim[f"dentro_{n}"] > 1).mean())
