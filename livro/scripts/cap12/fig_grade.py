"""Mapa de sensibilidade: Sharpe das 12 variantes, treino e teste."""
import sys; sys.path.insert(0, "..")
from estilo import *
from projeto import *

px, ret, cdi = carregar()
grades = []
for per in (TREINO, TESTE):
    g = pd.DataFrame(index=GRADE_J, columns=GRADE_K, dtype=float)
    for j in GRADE_J:
        for k in GRADE_K:
            r, _, _ = simular(alvo_momentum(px, j, k), ret, cdi, per)
            g.loc[j, k] = metricas(r, cdi)["Sharpe"]
    grades.append(g)
    print(per[0][:4]); print(g.round(2))
    print("min", g.min().min().round(2), "max", g.max().max().round(2),
          "mediana", round(g.stack().median(), 2))
c = grades[0].stack().corr(grades[1].stack(), method="spearman")
print("correlação de postos treino x teste:", round(c, 2))
razao = (grades[1] / grades[0]).stack()
print("teste/treino: de", round(razao.min(), 2), "a",
      round(razao.max(), 2))

fig, eixos = figura(altura=50, ncols=2)
nomes = ["Treino, 2013 a 2019", "Teste, 2020 a 2025"]
for ax, g, nome in zip(eixos, grades, nomes):
    ax.imshow(g.to_numpy(), cmap=SEQ_ROXO, vmin=0.2, vmax=1.15,
              aspect="auto")
    for i in range(len(GRADE_J)):
        for k in range(len(GRADE_K)):
            v = g.iloc[i, k]
            ax.text(k, i, f"{v:.2f}".replace(".", ","), ha="center",
                    va="center", fontsize=7.6,
                    color="white" if v > 0.72 else TINTA)
    ax.set_xticks(range(len(GRADE_K)), GRADE_K)
    ax.set_yticks(range(len(GRADE_J)), GRADE_J)
    ax.set_xlabel("K, número de ações na carteira")
    ax.set_title(nome)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # variante escolhida no treino: J = 12, K = 2
    ax.add_patch(mpl.patches.Rectangle((-0.5, 2.5), 1, 1, fill=False,
                                       ec=TINTA, lw=1.4))
eixos[0].set_ylabel("J, meses de formação")
salvar(fig, "cap12/grade")
