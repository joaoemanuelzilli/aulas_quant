"""AUC por dobra: quatro esquemas de validação, dois modelos."""
import sys; sys.path.insert(0, "..")
from estilo import *
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from l06_validacao import X, y, modelos, esquemas

fig, eixos = figura(altura=56, ncols=2, sharey=True, sharex=True)
nomes = list(esquemas)
for ax, (nome_m, m) in zip(eixos, modelos.items()):
    modelo = make_pipeline(StandardScaler(), m)
    for i, nome_e in enumerate(nomes):
        auc = cross_val_score(modelo, X, y, cv=esquemas[nome_e],
                              scoring="roc_auc")
        cor = OCRE if i == 0 else ROXO
        ax.plot(auc, [i] * len(auc), "o", color=cor, ms=3.2,
                alpha=0.45, markeredgewidth=0)
        ax.plot([auc.mean()], [i], "|", color=TINTA, ms=11,
                markeredgewidth=1.4)
        ax.text(0.765, i, f"{auc.mean():.3f}".replace(".", ","),
                va="center", ha="right", fontsize=7)
    ax.axvline(0.5, color=CINZA, lw=0.7, ls=(0, (4, 3)))
    titulos = {"logística": "Regressão logística",
               "15 vizinhos": "15 vizinhos mais próximos"}
    ax.set_title(titulos[nome_m])
    ax.set_xlabel("AUC em cada dobra de teste")
    ax.grid(False)
    ax.set_xlim(0.37, 0.77)
    ax.set_xticks([0.4, 0.5, 0.6])
    ax.text(0.765, -0.62, "média", ha="right", fontsize=6.6,
            color=CINZA)
    virgula(ax, "x", casas=1)
eixos[0].set_yticks(range(len(nomes)))
eixos[0].set_yticklabels(nomes)
eixos[0].set_ylim(len(nomes) - 0.5, -0.9)
for ax in eixos:
    ax.tick_params(axis="y", colors=TINTA)
salvar(fig, "cap09/validacao")
