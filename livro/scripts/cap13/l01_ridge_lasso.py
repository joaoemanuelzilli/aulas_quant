import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from painel import FEATS, dev_teste, painel

dev, _ = dev_teste(painel())
X = dev[FEATS].to_numpy()
X = (X - X.mean(0)) / X.std(0)             # features padronizadas
y = (dev["fwd"] - dev["fwd"].mean()).to_numpy() * 100   # em %
n, p = X.shape

# ridge pela fórmula fechada, conferida com o scikit-learn
lam = 2500.0
b_ridge = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
sk = Ridge(alpha=lam, fit_intercept=False).fit(X, y)
print("fórmula = sklearn:", np.allclose(b_ridge, sk.coef_))

b_mqo = np.linalg.solve(X.T @ X, X.T @ y)
# o Lasso do sklearn minimiza SQR/(2n) + alpha*|b|_1
lasso = Lasso(alpha=lam / (2 * n), fit_intercept=False)
b_lasso = lasso.fit(X, y).coef_
tab = pd.DataFrame({"MQO": b_mqo, "ridge": b_ridge,
                    "lasso": b_lasso}, index=FEATS)
print(tab.round(3).to_string())
print("norma L2:", np.linalg.norm(tab, axis=0).round(3))
