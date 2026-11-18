"""Três esquemas de validação para o mesmo LightGBM de fábrica:
o padrão do scikit-learn (cv=5, blocos sem embaralhar), k-fold
embaralhado e as divisões temporais com folga do capítulo."""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from painel import FEATS, cortes, dev_teste, painel
from modelos import lgbm_padrao

dev, _ = dev_teste(painel())
X, y = dev[FEATS].to_numpy(), dev["y"].to_numpy()
esquemas = {
    "cv=5 (padrão, sem embaralhar)": 5,
    "KFold(5, shuffle=True)": KFold(5, shuffle=True, random_state=0),
    "cortes (temporal, com folga)": cortes(dev.index),
}
for nome, cv in esquemas.items():
    auc = cross_val_score(lgbm_padrao(), X, y, cv=cv,
                          scoring="roc_auc")
    print(f"{nome:30s} média {auc.mean():.3f} | por dobra",
          auc.round(3))
