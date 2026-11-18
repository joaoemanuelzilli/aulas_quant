import numpy as np
from scipy.optimize import minimize

s_a, s_b, rho = 0.20, 0.30, 0.30
S = np.array([[s_a**2, rho * s_a * s_b],
              [rho * s_a * s_b, s_b**2]])
var = lambda w: w @ S @ w

# condições de 1a ordem: 2 S w - lambda 1 = 0 e w1 + w2 = c
def lagrange(c):
    M = np.block([[2 * S, -np.ones((2, 1))],
                  [np.ones((1, 2)), np.zeros((1, 1))]])
    return np.linalg.solve(M, [0, 0, c])   # (w1, w2, lambda)

w1, w2, lam = lagrange(1.0)
print(f"Lagrange: w = ({w1:.4f}, {w2:.4f})  lambda = {lam:.6f}")
print(f"variância mínima {var(np.array([w1, w2])):.6f}"
      f"  volatilidade {np.sqrt(var(np.array([w1, w2]))):.2%}")

soma = {"type": "eq", "fun": lambda w: w.sum() - 1}
res = minimize(var, x0=[0.5, 0.5], constraints=[soma],
               method="SLSQP", options={"ftol": 1e-12})
print(f"scipy:    w = ({res.x[0]:.4f}, {res.x[1]:.4f})")

# o multiplicador é a derivada do ótimo em relação a c
h = 1e-4
v = lambda c: var(lagrange(c)[:2])
dv = (v(1 + h) - v(1 - h)) / (2 * h)
print(f"dv/dc em c = 1 (numérica): {dv:.6f}")
