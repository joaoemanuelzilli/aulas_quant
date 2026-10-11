"""Testes de sanidade das fórmulas do Capítulo 4.  Uso: python teste_c04.py"""
import numpy as np

rng = np.random.default_rng(4)

# AR(1): média c/(1-phi), variância sigma²/(1-phi²), ACF phi^k
c, phi, T = 0.5, 0.8, 400_000
eps = rng.standard_normal(T)
y = np.empty(T); y[0] = c / (1 - phi)
for t in range(1, T):
    y[t] = c + phi * y[t - 1] + eps[t]
assert abs(y.mean() - 2.5) < 0.03 and abs(y.var() - 1 / 0.36) < 0.05
assert abs(np.corrcoef(y[2:], y[:-2])[0, 1] - phi**2) < 0.01

# MA(1): rho_1 = theta/(1+theta²), a mesma para theta e 1/theta
rho = lambda th: th / (1 + th**2)
assert abs(rho(0.7) - 0.4698) < 1e-4 and abs(rho(0.5) - rho(2.0)) < 1e-12 and rho(1.0) == 0.5

# GARCH(1,1): variância incondicional, curtose, previsão e meia-vida
w, a, b = 0.05, 0.075, 0.90
N, T = 4000, 3000
s2 = np.full(N, w / (1 - a - b)); e = np.empty((N, T))
for t in range(T):
    e[:, t] = np.sqrt(s2) * rng.standard_normal(N)
    s2 = w + a * e[:, t] ** 2 + b * s2
e = e[:, 500:]
assert abs(e.var() - w / (1 - a - b)) < 0.05                        # 2,0
curt = 3 * (1 - (a + b) ** 2) / (1 - (a + b) ** 2 - 2 * a**2)
assert abs((e**4).mean() / e.var() ** 2 - curt) < 0.25              # fórmula da curtose
vi, s1 = w / (1 - a - b), 2.546                                     # exemplo "um passo à mão"
assert abs(w + a * 16 + b * 1.44 - s1) < 1e-12
prev = s1
for h in range(2, 30):                                              # recursão x fórmula fechada
    prev = w + (a + b) * prev
    assert abs(prev - (vi + (a + b) ** (h - 1) * (s1 - vi))) < 1e-12
mv = np.log(0.5) / np.log(a + b)
assert abs((a + b) ** mv - 0.5) < 1e-12
# EWMA é o GARCH com omega = 0, alfa = 1 - lambda, beta = lambda
lam = 0.94
assert abs((lam * 1.44 + (1 - lam) * 16) - (0 + (1 - lam) * 16 + lam * 1.44)) < 1e-12
print("todos os testes passaram")
