import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from painel import FEATS, cortes, dev_teste, painel

dev, _ = dev_teste(painel())
X, y = dev[FEATS].to_numpy(), dev["y"].to_numpy()
modelos = {
    "logística": make_pipeline(StandardScaler(),
                               LogisticRegression()),
    "random forest": RandomForestClassifier(
        n_estimators=300, n_jobs=4, random_state=0),
    "xgboost": XGBClassifier(n_estimators=300, n_jobs=4,
                             random_state=0),
    "lightgbm": LGBMClassifier(n_estimators=300, n_jobs=4,
                               random_state=0, verbose=-1),
}
linhas = {}
for nome, m in modelos.items():
    auc = []
    for tr, va in cortes(dev.index):
        m.fit(X[tr], y[tr])
        auc.append([roc_auc_score(y[i],
                                  m.predict_proba(X[i])[:, 1])
                    for i in (tr, va)])
    auc = np.array(auc)
    linhas[nome] = {"AUC treino": auc[:, 0].mean(),
                    "AUC valid.": auc[:, 1].mean(),
                    "pior": auc[:, 1].min(),
                    "melhor": auc[:, 1].max()}
print(pd.DataFrame(linhas).T.round(3).to_string())
