import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# 1. máximo de x^(1/x) para x > 0
res = minimize_scalar(lambda x: -x ** (1 / x),
                      bounds=(0.5, 10), method="bounded")
print(f"argmax {res.x:.6f}  (e = {np.e:.6f})"
      f"  máximo {-res.fun:.6f}")

# 2. e^pi contra pi^e
print(f"e^pi = {np.e**np.pi:.4f}   pi^e = {np.pi**np.e:.4f}")

# 3. integral da gaussiana
val, err = quad(lambda x: np.exp(-x**2 / 2), -np.inf, np.inf)
print(f"integral {val:.10f}"
      f"   raiz de 2 pi {np.sqrt(2 * np.pi):.10f}")
