"""Modelos finais do capítulo, hiperparâmetros já escolhidos."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# configuração vencedora da busca aleatória (l06_busca.py)
MELHOR = dict(n_estimators=25, learning_rate=0.01, num_leaves=31,
              min_child_samples=20, subsample=0.5,
              colsample_bytree=0.7, reg_lambda=1)


def logistica():
    return make_pipeline(StandardScaler(), LogisticRegression())


def floresta(**kw):
    return RandomForestClassifier(n_estimators=300, n_jobs=4,
                                  random_state=0, **kw)


def lgbm_padrao():
    return LGBMClassifier(n_estimators=300, n_jobs=4,
                          random_state=0, verbose=-1)


def lgbm_ajustado():
    return LGBMClassifier(**MELHOR, subsample_freq=1, n_jobs=1,
                          random_state=0, verbose=-1)
