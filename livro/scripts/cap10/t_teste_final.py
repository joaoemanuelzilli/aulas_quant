"""Avaliação única no período intocado (2022-2025)."""
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from painel import FEATS, dev_teste, painel
import modelos

dev, teste = dev_teste(painel())
print("dev:", dev.index[0].date(), dev.index[-1].date(), len(dev),
      "| teste:", teste.index[0].date(), teste.index[-1].date(),
      len(teste), "| NaN no pivô:",
      teste.pivot(columns="acao", values="ret_seg").isna().sum().sum())
print("frequência de alta: dev", round(dev["y"].mean(), 4),
      "teste", round(teste["y"].mean(), 4))
linhas = {}
for nome in ("logistica", "floresta", "lgbm_padrao", "lgbm_ajustado"):
    m = getattr(modelos, nome)().fit(dev[FEATS], dev["y"])
    p_dev = m.predict_proba(dev[FEATS])[:, 1]
    p = m.predict_proba(teste[FEATS])[:, 1]
    linhas[nome] = {"AUC treino": roc_auc_score(dev["y"], p_dev),
                    "AUC teste": roc_auc_score(teste["y"], p),
                    "acurácia teste": accuracy_score(teste["y"], p > 0.5),
                    "frac. prev. alta": (p > 0.5).mean(),
                    "p min": p.min(), "p max": p.max()}
print(pd.DataFrame(linhas).T.round(3).to_string())

# incerteza da AUC do modelo escolhido na validação:
# reamostragem de blocos de 10 pregões seguidos
import numpy as np
m = modelos.lgbm_ajustado().fit(dev[FEATS], dev["y"])
teste = teste.assign(p=m.predict_proba(teste[FEATS])[:, 1])
dias = teste.index.unique()
blocos = [dias[i:i + 10] for i in range(0, len(dias), 10)]
rng = np.random.default_rng(0)
aucs = []
for _ in range(500):
    esc = rng.integers(0, len(blocos), len(blocos))
    a = teste.loc[np.concatenate([blocos[i] for i in esc])]
    aucs.append(roc_auc_score(a["y"], a["p"]))
n1 = teste["y"].sum(); n0 = len(teste) - n1
print("dp da AUC por blocos:", round(float(np.std(aucs)), 4),
      "| sob independência:",
      round(float(np.sqrt((n0 + n1 + 1) / (12 * n0 * n1))), 4))
