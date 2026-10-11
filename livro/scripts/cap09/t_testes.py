"""Testes de sanidade das funções centrais do Capítulo 9."""
import numpy as np
from features import carregar, montar, rsi, base
from l04_barreiras import tripla_barreira

d = carregar()
p = d["Close"]

# 1. RSI de Wilder contra um laço explícito
dif = p.diff().dropna().to_numpy()
u, q, n = np.clip(dif, 0, None), np.clip(-dif, 0, None), 14
mu, mq = u[:n].mean(), q[:n].mean()
ref = [100 * mu / (mu + mq)]
for i in range(n, len(dif)):
    mu, mq = (13 * mu + u[i]) / 14, (13 * mq + q[i]) / 14
    ref.append(100 * mu / (mu + mq))
assert np.allclose(rsi(p).dropna().to_numpy(), ref)

# 2. nenhuma feature usa o futuro (20 cortes)
cheio = montar(d)
for i in range(400, len(d), 180):
    assert np.allclose(montar(d.iloc[:i + 1]).iloc[-1],
                       cheio.iloc[i])

# 3. o alvo de base(h) é o log-retorno de t a t+h
X, y = base(h=5)
lp = np.log(p)
t = X.index[100]
k = p.index.get_loc(t)
assert np.isclose(y.loc[t], lp.iloc[k + 5] - lp.iloc[k])

# 4. tripla barreira contra um laço ingênuo
sigma = lp.diff().rolling(21).std()
b = tripla_barreira(p, sigma, h=10, b=1.0)
v = lp.to_numpy()
for t in range(21, len(v) - 10, 37):
    larg, rot, dias = np.sqrt(10) * sigma.iloc[t], 0, 10
    for j in range(1, 11):
        if abs(v[t + j] - v[t]) >= larg:
            rot, dias = np.sign(v[t + j] - v[t]), j
            break
    assert (b["rotulo"].iloc[t], b["dias"].iloc[t]) == (rot, dias)
assert b["rotulo"].iloc[-10:].isna().all()
print("todos os testes passaram")
