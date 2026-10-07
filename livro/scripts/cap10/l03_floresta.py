import numpy as np
from sklearn.ensemble import RandomForestClassifier
from painel import FEATS, cortes, dev_teste, painel

dev, _ = dev_teste(painel())
X, y = dev[FEATS].to_numpy(), dev["y"].to_numpy()
tr, va = cortes(dev.index)[-1]        # última divisão temporal


def corr_media(floresta, Xv):
    """Correlação média entre as previsões de pares de árvores."""
    P = np.array([a.predict_proba(Xv)[:, 1]
                  for a in floresta.estimators_])
    c = np.corrcoef(P)
    return c[np.triu_indices_from(c, k=1)].mean()


casos = (("bagging (m = p)", None), ("floresta (m = 3)", 3))
for nome, m in casos:
    rf = RandomForestClassifier(
        n_estimators=300, max_features=m, min_samples_leaf=50,
        oob_score=True, n_jobs=4, random_state=0)
    rf.fit(X[tr], y[tr])
    print(f"{nome}: correlação {corr_media(rf, X[va]):.3f}"
          f" | OOB {rf.oob_score_:.3f}"
          f" | validação {rf.score(X[va], y[va]):.3f}")

rf = RandomForestClassifier(n_estimators=300, oob_score=True,
                            n_jobs=4, random_state=0)
rf.fit(X[tr], y[tr])
print(f"árvores completas: treino {rf.score(X[tr], y[tr]):.3f}"
      f" | OOB {rf.oob_score_:.3f}"
      f" | validação {rf.score(X[va], y[va]):.3f}")
