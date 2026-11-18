import numpy as np
from scipy.optimize import minimize

s_a, s_b, rho = 0.20, 0.30, 0.80     # correlação alta
S = np.array([[s_a**2, rho * s_a * s_b],
              [rho * s_a * s_b, s_b**2]])
var = lambda w: w @ S @ w
soma = {"type": "eq", "fun": lambda w: w.sum() - 1}


def otimo(piso):
    """Mínima variância com soma 1 e cada peso >= piso."""
    r = minimize(var, [0.5, 0.5], constraints=[soma],
                 bounds=[(piso, None)] * 2, method="SLSQP",
                 options={"ftol": 1e-14})
    return r.x


livre = otimo(None)
w = otimo(0.0)
print(f"sem restrição de sinal: w = {np.round(livre, 4)}")
print(f"com pesos >= 0:         w = {np.round(w, 4)}")

g = 2 * S @ w                 # gradiente da variância no ótimo
lam = g[0]                    # w_A > 0: restrição inativa
mu = g[1] - lam               # w_B = 0: restrição ativa
print(f"lambda = {lam:.4f}   mu = {mu:.4f}"
      f"   mu * w_B = {mu * w[1]:.1e}")

h = 1e-3                      # permite vender h a descoberto
dv = (var(w) - var(otimo(-h))) / h
print(f"queda da variância por unidade de piso: {dv:.4f}")
