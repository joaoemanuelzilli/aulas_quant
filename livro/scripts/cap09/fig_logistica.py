"""Sigmoide e curva ROC da logística no período de teste."""
import sys; sys.path.insert(0, "..")
from estilo import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import base

X, ret = base(h=1)
y = (ret > 0).astype(int)
tr = X.index < "2019"
m = make_pipeline(StandardScaler(), LogisticRegression())
m.fit(X[tr], y[tr])
prob = m.predict_proba(X[~tr])[:, 1]
fpr, tpr, _ = roc_curve(y[~tr], prob)
auc = roc_auc_score(y[~tr], prob)

fig, (a1, a2) = figura(altura=53, ncols=2)
z = np.linspace(-6, 6, 300)
a1.plot(z, 1 / (1 + np.exp(-z)), color=ROXO)
a1.axhline(0.5, color=CINZA, lw=0.6, ls=(0, (4, 3)))
a1.axvline(0, color=CINZA_CLARO, lw=0.5, zorder=0)
a1.plot([0], [0.5], "o", color=OCRE, ms=3.5)
a1.annotate("$z=0$: chances iguais", (0, 0.5), xytext=(6, -11),
            textcoords="offset points", fontsize=6.8)
a1.set_xlabel("$z = \\beta_0 + \\mathbf{x}^{\\mathsf{T}}"
              "\\boldsymbol{\\beta}$ (log-chances)")
a1.set_ylabel("probabilidade $\\sigma(z)$")
a1.set_title("Função sigmoide")
a1.set_ylim(-0.02, 1.02)
virgula(a1)

a2.plot([0, 1], [0, 1], color=CINZA, lw=0.7, ls=(0, (4, 3)))
a2.plot(fpr, tpr, color=ROXO, lw=1.1)
a2.set_xlabel("taxa de falsos positivos")
a2.set_ylabel("taxa de verdadeiros positivos (recall)")
a2.set_title("Curva ROC no teste")
a2.text(0.97, 0.06, f"AUC = {auc:.3f}".replace(".", ",")
        + "\n(o acaso dá 0,5)", ha="right", fontsize=7)
a2.set_aspect("equal")
a2.grid(False)
virgula(a2); virgula(a2, "x")
salvar(fig, "cap09/logistica")
print(f"AUC {auc:.4f}; prob min/max {prob.min():.3f} "
      f"{prob.max():.3f}; quantis 5/95 "
      f"{np.quantile(prob, [0.05, 0.95]).round(3)}")
