"""Dependência parcial do LightGBM ajustado em duas features: a curva
do modelo final e as curvas dos modelos de cada janela de treino."""
import sys; sys.path.insert(0, "..")
from estilo import *
from sklearn.inspection import partial_dependence
from painel import FEATS, cortes, dev_teste, painel
from modelos import lgbm_ajustado

dev, _ = dev_teste(painel())
X, y = dev[FEATS], dev["y"]
pares = ["vol_razao", "ret_5"]
fig, eixos = figura(altura=54, ncols=2, sharey=True)
final = lgbm_ajustado().fit(X, y)
for ax, nome in zip(eixos, pares):
    grade = np.quantile(X[nome], np.linspace(0.02, 0.98, 49))
    def curva(m, base):
        d = partial_dependence(m, base, [nome], custom_values={
            nome: grade}, kind="average")
        return d["average"][0]
    for k, (tr, _) in enumerate(cortes(dev.index)):
        m = lgbm_ajustado().fit(X.iloc[tr], y.iloc[tr])
        c = curva(m, X.iloc[tr])
        ax.plot(grade, c, color=CINZA_CLARO, lw=0.9)
        print(nome, "janela", k, "amplitude", round(np.ptp(c), 4),
              "ini", round(c[0], 3), "fim", round(c[-1], 3))
    c = curva(final, X)
    ax.plot(grade, c, color=ROXO)
    print(nome, "final: de", round(c.min(), 3), "a", round(c.max(), 3))
    ax.axhline(y.mean(), color=TINTA, lw=0.6, ls=(0, (4, 3)))
    ax.plot(np.quantile(X[nome], np.linspace(0.1, 0.9, 9)),
            np.full(9, 0.4718), "|", color=CINZA, ms=5)
    ax.set_title(f"Dependência parcial em {nome}")
    ax.set_xlabel(nome)
    ax.set_ylim(0.47, 0.55)
    ax.set_yticks(np.arange(0.48, 0.55, 0.02))
    virgula(ax, casas=2)
eixos[0].set_ylabel("probabilidade média prevista")
eixos[0].set_xticks([0.6, 0.9, 1.2, 1.5, 1.8])
virgula(eixos[0], "x", 1)
pct(eixos[1], "x")
print("frequência de alta no desenvolvimento:", round(y.mean(), 4))
salvar(fig, "cap10/dependencia")
