"""Gabaritos numéricos dos exercícios do Capítulo 5."""
import runpy

import numpy as np

from calculo import (bs, convexidade, duration_mod,
                     gradiente_descendente, gregas, preco_ltn)

# 5.1 LTN de 756 du a 12%, choque de +200 pb
P = preco_ltn(0.12, 756)
D, C = duration_mod(0.12, 756), convexidade(0.12, 756)
dy = 0.02
ex = preco_ltn(0.14, 756) / P - 1
print(f"5.1  P = {P:.4f}  D = {D:.4f}  C = {C:.4f}  "
      f"1a {-D*dy:+.4%}  2a {-D*dy + C*dy**2/2:+.4%}  exata {ex:+.4%}")

# 5.2 log-retorno da LTN do capítulo para +10 pb
D0 = duration_mod(0.1436, 532)
exato = np.log(preco_ltn(0.1446, 532) / preco_ltn(0.1436, 532))
print(f"5.2  aprox {-D0 * 0.001:+.5f}  exato {exato:+.5f}")

# 5.3 gradiente e hessiana com rho = -0,5
S = np.array([[0.04, -0.03], [-0.03, 0.09]])
w = np.array([0.5, 0.5])
H = 2 * S
print(f"5.3  grad = {2 * S @ w}  H = {H.tolist()}  "
      f"det = {np.linalg.det(H):.4f}")

# 5.4 título com dois fluxos
t = np.array([1.0, 2.0]); fl = np.array([100.0, 1100.0])
vp = fl / 1.14**t
Pc = vp.sum(); mac = (t * vp).sum() / Pc
dP = -(t * fl / 1.14 ** (t + 1)).sum()
print(f"5.4  P = {Pc:.4f}  Macaulay = {mac:.4f}  "
      f"modificada = {-dP / Pc:.4f}  Mac/(1+y) = {mac / 1.14:.4f}")

# 5.5 gradiente descendente com retornos em fração
ns = runpy.run_path("l03_gradiente.py")
x, y = ns["x"] / 100, ns["y"] / 100
grad = lambda t: np.array([-2 * (y - t[0] - t[1] * x).mean(),
                           -2 * ((y - t[0] - t[1] * x) * x).mean()])
Hf = 2 * np.array([[1, x.mean()], [x.mean(), (x**2).mean()]])
lim = 2 / np.linalg.eigvalsh(Hf).max()
b_alvo = np.polyfit(x, y, 1)[0]
tt = np.array([0.0, 0.0]); k = 0
while abs(tt[1] - b_alvo) >= 0.5e-4:
    tt = tt - 0.95 * lim * grad(tt); k += 1
print(f"5.5  limite eta < {lim:.4f};  passos até 4 casas em b "
      f"com eta = 0,95 x limite: {k}")

# 5.6 Lagrange com rho = -0,5
S = np.array([[0.04, -0.03], [-0.03, 0.09]])
M = np.block([[2 * S, -np.ones((2, 1))],
              [np.ones((1, 2)), np.zeros((1, 1))]])
w1, w2, lam = np.linalg.solve(M, [0, 0, 1])
print(f"5.6  w = ({w1:.4f}, {w2:.4f})  lambda = {lam:.6f}  "
      f"vol = {np.sqrt(lam / 2):.4%};  ativa se rho > {0.2/0.3:.4f}")

# 5.7 volatilidade implícita por Newton
s, it = 0.30, 0
while True:
    novo = s - (bs(30, 30, 0.25, 0.14, s) - 2.50) \
        / gregas(30, 30, 0.25, 0.14, s)["vega"]
    it += 1
    if abs(novo - s) < 1e-7:
        break
    s = novo
print(f"5.7  sigma implícita = {novo:.6f} em {it} iterações")

# 5.9 n^(1/n) nos inteiros
v = {n: n ** (1 / n) for n in range(1, 8)}
print(f"5.9  2^(1/2) = {v[2]:.4f}  3^(1/3) = {v[3]:.4f}  "
      f"melhor n = {max(v, key=v.get)}")
