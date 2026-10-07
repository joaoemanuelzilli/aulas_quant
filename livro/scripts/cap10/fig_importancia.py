"""MDI (dentro da amostra) contra importância por permutação fora da
amostra, com uma feature de ruído puro acrescentada ao painel."""
import sys; sys.path.insert(0, "..")
from estilo import *
from sklearn.inspection import permutation_importance
from painel import FEATS, cortes, dev_teste, painel
from modelos import floresta

dev, _ = dev_teste(painel())
rng = np.random.default_rng(0)
dev = dev.assign(ruido=rng.normal(size=len(dev)))
cols = FEATS + ["ruido"]
X, y = dev[cols].to_numpy(), dev["y"].to_numpy()
tr, va = cortes(dev.index)[-1]

rf = floresta().fit(X[tr], y[tr])
mdi = pd.Series(rf.feature_importances_, cols)
dp_mdi = pd.Series(np.std([a.feature_importances_
                           for a in rf.estimators_], axis=0), cols)
perm = permutation_importance(rf, X[va], y[va], scoring="roc_auc",
                              n_repeats=30, random_state=0, n_jobs=4)
mda = pd.Series(perm.importances_mean, cols)
dp = pd.Series(perm.importances_std, cols)
from sklearn.metrics import roc_auc_score
print("AUC de validação:",
      round(roc_auc_score(y[va], rf.predict_proba(X[va])[:, 1]), 4))
print(pd.DataFrame({"MDI": mdi, "perm": mda, "dp perm": dp})
      .round(4).to_string())
print("perm / dp:", (mda / dp).round(1).to_dict())
print("1/p =", round(1 / len(cols), 4),
      "| features com perm > 2 dp:", list(mda[mda > 2 * dp].index))

ordem = mdi.sort_values().index
cores = [OCRE if c == "ruido" else ROXO for c in ordem]
fig, (a1, a2) = figura(altura=60, ncols=2, sharey=True)
a1.barh(ordem, mdi[ordem], color=cores, height=0.62)
a1.axvline(1 / len(cols), color=TINTA, lw=0.6, ls=(0, (4, 3)))
a1.text(1 / len(cols), 10.75, "$1/p$", fontsize=6.8, ha="center")
a1.set_title("MDI (calculada no treino)", pad=12)
a1.set_xlabel("redução média de impureza")
pct(a1, "x")
a2.barh(ordem, mda[ordem], xerr=2 * dp[ordem], color=cores,
        height=0.62, error_kw=dict(lw=0.6, ecolor=TINTA))
a2.axvline(0, color=TINTA, lw=0.6)
a2.set_title("Permutação (na validação)", pad=12)
a2.set_xlabel("queda da AUC ao embaralhar")
a2.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: f"{v:.2f}".replace(".", ",")))
for a in (a1, a2):
    a.grid(False)
    a.tick_params(axis="y", labelsize=7, labelcolor=TINTA)
salvar(fig, "cap10/importancia")
