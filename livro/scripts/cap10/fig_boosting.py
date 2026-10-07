"""Perda de treino e de validação do LightGBM por número de árvores,
para duas taxas de aprendizado (painel real, última divisão temporal)."""
import sys; sys.path.insert(0, "..")
from estilo import *
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from painel import FEATS, cortes, dev_teste, painel

dev, _ = dev_teste(painel())
X, y = dev[FEATS].to_numpy(), dev["y"].to_numpy()
tr, va = cortes(dev.index)[-1]
base = log_loss(y[va], np.full(len(va), y[tr].mean()))
print("perda de referência (prever a frequência do treino):",
      round(base, 4))

M = 400
fig, eixos = figura(altura=52, ncols=2, sharey=True)
for ax, eta in zip(eixos, (0.3, 0.02)):
    m = LGBMClassifier(n_estimators=M, learning_rate=eta,
                       num_leaves=15, random_state=0, verbose=-1)
    m.fit(X[tr], y[tr], eval_X=(X[tr], X[va]), eval_y=(y[tr], y[va]),
          eval_metric="binary_logloss")
    lt = np.array(m.evals_result_["valid_0"]["binary_logloss"])
    lv = np.array(m.evals_result_["valid_1"]["binary_logloss"])
    k = lv.argmin()
    print(f"eta={eta}: mínimo da validação {lv[k]:.4f} com {k + 1} "
          f"árvores; com {M}: treino {lt[-1]:.4f} validação {lv[-1]:.4f}")
    n = np.arange(1, M + 1)
    ax.plot(n, lt, color=CINZA)
    ax.plot(n, lv, color=ROXO)
    ax.axhline(base, color=TINTA, lw=0.6, ls=(0, (4, 3)))
    ax.plot([k + 1], [lv[k]], "o", color=ROXO, ms=3.5)
    ax.set_title(f"Taxa de aprendizado {eta}".replace(".", ","))
    ax.set_xlabel("número de árvores")
    ax.set_ylim(0.3, 1.0)
    virgula(ax, casas=1)
    ax.annotate("treino", (200, lt[199]), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top",
                fontsize=7.2)
    ax.annotate("validação", (M, lv[-1]), xytext=(-3, 6),
                textcoords="offset points", ha="right", fontsize=7.2)
eixos[0].set_ylabel("perda logarítmica")
salvar(fig, "cap10/boosting")
