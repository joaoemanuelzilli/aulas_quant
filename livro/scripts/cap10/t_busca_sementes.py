"""Repete a busca aleatória de l06_busca.py com seis sementes no
sorteio das configurações. Não consulta o período final."""
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from painel import FEATS, cortes, dev_teste, painel

dev, _ = dev_teste(painel())
X, y = dev[FEATS].to_numpy(), dev["y"].to_numpy()
grade = {
    "n_estimators": [25, 50, 100, 200, 400],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "num_leaves": [3, 5, 9, 15, 31],
    "min_child_samples": [20, 100, 300, 1000],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda": [0, 1, 10, 100],
}
for semente in range(6):
    base = LGBMClassifier(subsample_freq=1, n_jobs=1,
                          random_state=0, verbose=-1)
    b = RandomizedSearchCV(
        base, grade, n_iter=40, scoring="roc_auc",
        cv=cortes(dev.index), n_jobs=6, random_state=semente)
    b.fit(X, y)
    m = b.best_params_
    print(f"semente {semente}: AUC {b.best_score_:.4f} | árvores "
          f"{m['n_estimators']}, folhas {m['num_leaves']}, taxa "
          f"{m['learning_rate']}, reg_lambda {m['reg_lambda']}, "
          f"subsample {m['subsample']}")
