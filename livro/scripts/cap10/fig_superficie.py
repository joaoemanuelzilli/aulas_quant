"""Probabilidade prevista por uma árvore rasa, uma árvore completa e
uma floresta, usando só duas features (para caber no plano)."""
import sys; sys.path.insert(0, "..")
from estilo import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from painel import dev_teste, painel

dev, _ = dev_teste(painel())
F2 = ["ret_21", "vol_razao"]
X, y = dev[F2].to_numpy(), dev["y"].to_numpy()
lim = np.percentile(X, [1, 99], axis=0)
g1, g2 = np.meshgrid(np.linspace(*lim[:, 0], 220),
                     np.linspace(*lim[:, 1], 220))
G = np.c_[g1.ravel(), g2.ravel()]
modelos = {
    "Árvore de profundidade 2":
        DecisionTreeClassifier(max_depth=2, random_state=0),
    "Árvore completa":
        DecisionTreeClassifier(random_state=0),
    "Floresta, folhas com 100 obs.":
        RandomForestClassifier(300, min_samples_leaf=100, n_jobs=4,
                               random_state=0),
}
fig, eixos = figura(altura=58, largura="larga", ncols=3, sharey=True)
for ax, (nome, m) in zip(eixos, modelos.items()):
    m.fit(X, y)
    z = m.predict_proba(G)[:, 1].reshape(g1.shape)
    print(nome, "| folhas:", getattr(m, "get_n_leaves", lambda: "-")(),
          "| acurácia de treino:", round(m.score(X, y), 3))
    im = ax.pcolormesh(g1, g2, z, cmap=DIV, vmin=0, vmax=1,
                       rasterized=True)
    ax.set_title(nome)
    ax.set_xlabel("ret_21")
    ax.grid(False)
    pct(ax, "x")
    virgula(ax, casas=1)
eixos[0].set_ylabel("vol_razao")
cb = fig.colorbar(im, ax=eixos, shrink=0.8, pad=0.02, aspect=18)
cb.set_label("probabilidade prevista de alta")
cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: f"{v:.1f}".replace(".", ",")))
salvar(fig, "cap10/superficie")
