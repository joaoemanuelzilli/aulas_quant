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

# o que o puro acaso produz dentro da amostra: embaralhar o alvo
rng = np.random.default_rng(0)
Xt, yt = X[treino].to_numpy(), y[treino].to_numpy()
acaso = []
for _ in range(2000):
    yp = rng.permutation(yt)
    acaso.append(LinearRegression().fit(Xt, yp).score(Xt, yp))
print(f"R2 dentro com alvo embaralhado: mediana "
      f"{np.median(acaso):.4%}, percentil 95 "
      f"{np.quantile(acaso, 0.95):.4%}")
obtido = modelo.score(X[treino], y[treino])
print(f"embaralhamentos com R2 maior que o obtido: "
      f"{np.mean(np.array(acaso) > obtido):.0%}")

# incerteza do R2 fora: reamostrar o teste em blocos de 21 dias
ye, n, B = y[teste].to_numpy(), teste.sum(), 21
inicios = np.arange(0, n - B + 1)
r2b = []
for _ in range(2000):
    i = rng.choice(inicios, n // B)
    idx = (i[:, None] + np.arange(B)).ravel()
    r2b.append(r2_fora(ye[idx], prev[idx], 0.0))
r2b = np.array(r2b)
print(f"R2 fora (ref. zero) em 2000 reamostras: "
      f"desvio-padrão {r2b.std():.4%}")
print(f"reamostras com R2 fora negativo: {(r2b < 0).mean():.0%}")
