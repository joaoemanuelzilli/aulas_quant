import numpy as np
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
base = LGBMClassifier(subsample_freq=1, n_jobs=1, random_state=0,
                      verbose=-1)
busca = RandomizedSearchCV(
    base, grade, n_iter=40, scoring="roc_auc",
    cv=cortes(dev.index), n_jobs=4, random_state=0).fit(X, y)

res = pd.DataFrame(busca.cv_results_)["mean_test_score"]
m = busca.best_params_
print("melhor configuração:")
print({k: m[k] for k in list(m)[:4]})
print({k: m[k] for k in list(m)[4:]})
print(f"AUC de validação das 40 tentativas:\n"
      f"média {res.mean():.4f} | pior {res.min():.4f} | "
      f"melhor {res.max():.4f}")
