import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import base

X, ret = base(h=1)
y = (ret > 0).astype(int)         # 1 = alta amanhã, 0 = queda
treino = X.index < "2019"
teste = ~treino

# o Pipeline ajusta média e desvio do padronizador só no treino
modelo = make_pipeline(StandardScaler(), LogisticRegression())
modelo.fit(X[treino], y[treino])
prob = modelo.predict_proba(X[teste])[:, 1]   # P(alta)
prev = (prob > 0.5).astype(int)               # limiar de decisão

print("matriz de confusão (linhas: real 0, 1):")
print(confusion_matrix(y[teste], prev))
print(f"acurácia  {accuracy_score(y[teste], prev):.4f}")
print(f"precisão  {precision_score(y[teste], prev):.4f}")
print(f"recall    {recall_score(y[teste], prev):.4f}")
print(f"F1        {f1_score(y[teste], prev):.4f}")
print(f"AUC       {roc_auc_score(y[teste], prob):.4f}")
print(f"sempre alta (classe majoritária): {y[teste].mean():.4f}")
sinal = (X.loc[teste, "ret_1"] > 0).astype(int)
print(f"repetir o sinal de hoje: "
      f"{accuracy_score(y[teste], sinal):.4f}")
modulo = np.abs(np.exp(ret[teste]) - 1).mean()
print(f"variação diária média, em módulo: {modulo:.4f}")
coef = pd.DataFrame({"beta": modelo[-1].coef_[0]},
                    index=X.columns)
coef["razão de chances"] = np.exp(coef["beta"])
print(coef.T.iloc[:, :5].round(3).to_string())
print(coef.T.iloc[:, 5:].round(3).to_string())
