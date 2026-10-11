"""Confere as contas feitas à mão no texto do Capítulo 9."""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# quatro pontos: reta de mínimos quadrados
x = np.array([-1.0, 0.0, 1.0, 2.0])
y = np.array([-1.5, 0.5, 1.0, 3.0])
sxy = ((x - x.mean()) * (y - y.mean())).sum()
sxx = ((x - x.mean()) ** 2).sum()
b1 = sxy / sxx
b0 = y.mean() - b1 * x.mean()
res = y - (b0 + b1 * x)
sqt = ((y - y.mean()) ** 2).sum()
print("médias", x.mean(), y.mean(), "| Sxy", sxy, "Sxx", sxx)
print("inclinação", b1, "intercepto", round(b0, 4))
print("previsto", (b0 + b1 * x).round(2))
print("resíduos", res.round(2), "soma", round(res.sum(), 10))
sqr = res @ res
print("SQR", round(sqr, 4), "SQT", sqt, "R2", round(1 - sqr / sqt, 4))
X = np.column_stack([np.ones(4), x])
beta = np.linalg.solve(X.T @ X, X.T @ y)
print("forma matricial:", beta.round(4))

# AUC com cinco dias
nota = [0.62, 0.55, 0.48, 0.58, 0.41]
real = [1, 1, 1, 0, 0]
print("AUC dos cinco dias:", round(roc_auc_score(real, nota), 4))

# logística: do escore à probabilidade
sig = lambda z: 1 / (1 + np.exp(-z))
print("p(x=2), p(x=-2):", sig(0.1 + 0.25 * 2).round(3),
      sig(0.1 - 0.25 * 2).round(3))
print("efeito máximo de beta=0,25:", 0.25 / 4,
      "| em p=0,9:", round(0.25 * 0.9 * 0.1, 4))

# acertar 53% das vezes
p, m = 0.53, 0.0164
g = (2 * p - 1) * m
print(f"ganho diário {g:.4%} | em 252 dias {252 * g:.1%}")
print("E|r|/sigma da normal:", round(np.sqrt(2 / np.pi), 3))
sd = np.sqrt(2 / np.pi) * (2 * p - 1)
print(f"Sharpe diário {sd:.3f} | anual {sd * np.sqrt(252):.2f}")
print("R$ 10 mil: ganho", round(10000 * g, 2),
      "custo", 10000 * 0.001)
print("acaso com 50 features e 500 obs.:", round(50 / 499, 3))
