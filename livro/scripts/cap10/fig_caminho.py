"""Caminho dos coeficientes de ridge e lasso e escolha de lambda
por validação temporal (retorno de 5 dias, em %, sobre o painel)."""
import sys; sys.path.insert(0, "..")
from estilo import *
from sklearn.linear_model import Lasso, Ridge, lasso_path
from painel import FEATS, cortes, dev_teste, painel

dev, _ = dev_teste(painel())
Xb = dev[FEATS].to_numpy()
yb = dev["fwd"].to_numpy() * 100
X = (Xb - Xb.mean(0)) / Xb.std(0)
y = yb - yb.mean()
n, p = X.shape

# mesma definição nos dois: SQR + lam * penalidade. O scikit-learn
# escreve o lasso como SQR/(2n) + alpha*|b|_1, logo alpha = lam/(2n).
lams = np.logspace(1, 7, 80)
coef_r = np.array([np.linalg.solve(X.T @ X + l * np.eye(p), X.T @ y)
                   for l in lams])
_, coef_l, _ = lasso_path(X, y, alphas=lams / (2 * n))
coef_l = coef_l.T[::-1]       # lasso_path devolve lambda decrescente


def r2_fora(lam, tipo):
    """R² fora da amostra contra a média do treino, por divisão.
    O lasso recebe alpha = lam / (2 * n_treino): o n é o de cada
    treino, para que lam tenha o significado da Equação do texto."""
    sq, sq0 = 0.0, 0.0
    for tr, va in cortes(dev.index):
        mu, sd = Xb[tr].mean(0), Xb[tr].std(0)
        if tipo == "lasso":
            modelo = Lasso(alpha=lam / (2 * len(tr)),
                           fit_intercept=False)
        else:
            modelo = Ridge(alpha=lam, fit_intercept=False)
        m = modelo.fit((Xb[tr] - mu) / sd, yb[tr] - yb[tr].mean())
        prev = m.predict((Xb[va] - mu) / sd) + yb[tr].mean()
        sq += np.sum((yb[va] - prev) ** 2)
        sq0 += np.sum((yb[va] - yb[tr].mean()) ** 2)
    return 1 - sq / sq0


val_l = np.array([r2_fora(a, "lasso") for a in lams])
val_r = np.array([r2_fora(a, "ridge") for a in lams])
il, ir = val_l.argmax(), val_r.argmax()
print(f"lasso: melhor lambda {lams[il]:.0f}  R2 fora {val_l[il]:.5f}"
      f"  coef. não nulos {np.sum(coef_l[il] != 0)}")
print(f"ridge: melhor lambda {lams[ir]:.0f}  R2 fora {val_r[ir]:.5f}")
print(f"lambda = {lams[0]:.0f}: R2 fora lasso {val_l[0]:.5f}"
      f"  ridge {val_r[0]:.5f}")
print("lasso, ordem em que zera:", [FEATS[j] for j in np.argsort(
    [np.argmax(coef_l[:, j] == 0) for j in range(p)])])
print("lasso zera tudo a partir de lambda =",
      round(float(2 * np.max(np.abs(X.T @ y)))))
b_mqo = np.linalg.solve(X.T @ X, X.T @ y)
print("n =", n, " R2 dentro da amostra do MQO:",
      round(1 - np.sum((y - X @ b_mqo) ** 2) / np.sum(y ** 2), 5))
print("R2 fora do MQO:",
      round(r2_fora(0.0, "ridge"), 5))

destaque = {"vol_21": ROXO, "vol_razao": OCRE, "rsi_14": VERDE,
            "pm_50": AZUL}
fig, (a1, a2) = figura(altura=60, ncols=2, sharey=True)
for ax, coef in ((a1, coef_r), (a2, coef_l)):
    for j, nome in enumerate(FEATS):
        cor = destaque.get(nome, CINZA_CLARO)
        ax.plot(lams, coef[:, j], color=cor,
                lw=1.25 if nome in destaque else 0.8,
                zorder=3 if nome in destaque else 2)
    ax.set_xscale("log")
    ax.axhline(0, color=TINTA, lw=0.6)
    ax.set_xlabel("$\\lambda$ (escala log)")
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    virgula(ax, casas=1)
a1.set_title("Ridge: encolhe todos")
a2.set_title("Lasso: zera um a um")
a1.set_ylabel("coeficiente (feature padronizada)")
k = 6
for nome, dy, va in (("vol_21", 3, "bottom"), ("vol_razao", -4, "top"),
                     ("rsi_14", 3, "bottom"), ("pm_50", 2, "bottom")):
    j = FEATS.index(nome)
    a1.annotate(nome, (lams[k], coef_r[k, j]), xytext=(4, dy),
                textcoords="offset points", fontsize=6.8, va=va)
for ax, i in ((a1, ir), (a2, il)):
    ax.axvline(lams[i], color=TINTA, lw=0.6, ls=(0, (4, 3)))
a2.annotate("$\\lambda$ escolhido na\nvalidação temporal",
            (lams[il], 0.78), xytext=(5, 0),
            textcoords="offset points", fontsize=6.8, va="center")
salvar(fig, "cap10/caminho")
