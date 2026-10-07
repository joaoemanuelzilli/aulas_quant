import numpy as np
from sklearn.linear_model import LinearRegression
from features import base

X, y = base(h=1)                  # y = log-retorno de t a t+1
treino = X.index < "2019"
teste = ~treino

modelo = LinearRegression().fit(X[treino], y[treino])
prev = modelo.predict(X[teste])


def r2_fora(y, prev, referencia):
    """1 - SQE do modelo / SQE da previsão de referência."""
    sqe = ((y - prev) ** 2).sum()
    return 1 - sqe / ((y - referencia) ** 2).sum()


# média histórica até t-1, atualizada a cada dia
media = y.expanding().mean().shift(1)[teste]
print(f"treino: {treino.sum()} dias | teste: {teste.sum()} dias")
print(f"R2 dentro da amostra:      "
      f"{modelo.score(X[treino], y[treino]):8.4%}")
print(f"R2 fora, ref. média hist.: "
      f"{r2_fora(y[teste], prev, media):8.4%}")
print(f"R2 fora, ref. zero:        "
      f"{r2_fora(y[teste], prev, 0.0):8.4%}")
print("R2 fora por ano (ref. zero):")
for ano in range(2019, 2026):
    a = y[teste].index.year == ano
    print(f"  {ano}: {r2_fora(y[teste][a], prev[a], 0.0):8.4%}")
