import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.5, 200)
X = x[:, None]

eta, M = 0.1, 100
F = np.full(len(y), y.mean())         # F_0: a melhor constante
for m in range(1, M + 1):
    residuo = y - F                  # gradiente negativo da perda
    arvore = DecisionTreeRegressor(max_depth=2)
    F += eta * arvore.fit(X, residuo).predict(X)
    if m in (1, 10, 100):
        print(f"{m:3d} árvores: EQM de treino "
              f"{np.mean((y - F) ** 2):.4f}")

gbr = GradientBoostingRegressor(
    learning_rate=eta, n_estimators=M, max_depth=2,
    criterion="squared_error", random_state=0).fit(X, y)
print("igual ao scikit-learn:", np.allclose(F, gbr.predict(X)))
