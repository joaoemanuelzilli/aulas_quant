"""Viés, variância e erro de treino/teste por complexidade.

Experimento simulado: árvores de regressão de profundidade crescente
ajustadas a 300 amostras de treino diferentes do mesmo processo.
"""
import sys; sys.path.insert(0, "..")
from estilo import *
from sklearn.tree import DecisionTreeRegressor

N, REPS, PROFS = 200, 300, range(1, 11)


def f(x):
    return np.sin(2 * np.pi * x)


def experimento(sigma, semente=0):
    rng = np.random.default_rng(semente)
    x0 = np.linspace(0.005, 0.995, 200)
    linhas = []
    for prof in PROFS:
        prev = np.empty((REPS, x0.size))
        treino = np.empty(REPS)
        for b in range(REPS):
            x = rng.uniform(0, 1, N)
            y = f(x) + rng.normal(0, sigma, N)
            m = DecisionTreeRegressor(max_depth=prof).fit(x[:, None], y)
            prev[b] = m.predict(x0[:, None])
            treino[b] = np.mean((m.predict(x[:, None]) - y) ** 2)
        y0 = f(x0) + rng.normal(0, sigma, prev.shape)
        linhas.append(dict(
            prof=prof, treino=treino.mean(),
            teste=np.mean((y0 - prev) ** 2),
            vies2=np.mean((prev.mean(0) - f(x0)) ** 2),
            var=prev.var(0).mean()))
    return pd.DataFrame(linhas).set_index("prof")


SIGMA = 0.5
r = experimento(SIGMA)
r["soma"] = r["vies2"] + r["var"] + SIGMA ** 2
print(r.round(4).to_string())
print("profundidade de menor erro de teste:", r["teste"].idxmin())
for s in (0.25, 1.0, 2.0):
    print(f"sigma={s}: ótimo em", experimento(s)["teste"].idxmin())

fig, (a1, a2) = figura(altura=54, ncols=2)
a1.plot(r.index, r["treino"], color=CINZA, marker="o", ms=2.5)
a1.plot(r.index, r["teste"], color=ROXO, marker="o", ms=2.5)
a1.axhline(SIGMA ** 2, color=TINTA, lw=0.6, ls=(0, (4, 3)))
a1.text(10, SIGMA ** 2 + 0.008, "ruído irredutível $\\sigma^2$",
        fontsize=6.8, va="bottom", ha="right")
a1.annotate("erro de teste", (10, r["teste"][10]), xytext=(-2, 6),
            textcoords="offset points", ha="right", fontsize=7.2)
a1.annotate("erro de treino", (6, r["treino"][6]), xytext=(-4, -5),
            textcoords="offset points", ha="right", va="top",
            fontsize=7.2)
a1.set_title("Erro quadrático médio")
a2.plot(r.index, r["vies2"], color=OCRE, marker="o", ms=2.5)
a2.plot(r.index, r["var"], color=VERDE, marker="o", ms=2.5)
a2.annotate("viés²", (1, r["vies2"][1]), xytext=(5, 5),
            textcoords="offset points", fontsize=7.2)
a2.annotate("variância", (10, r["var"][10]), xytext=(-4, 6),
            textcoords="offset points", ha="right", fontsize=7.2)
a2.set_title("Decomposição do erro de teste")
for a in (a1, a2):
    a.set_xlabel("profundidade máxima da árvore")
    a.set_xticks(range(1, 11, 1))
    a.set_ylim(0, 0.5)
    virgula(a, casas=1)
salvar(fig, "cap10/vies_variancia")
