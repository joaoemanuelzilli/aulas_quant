from itertools import product

import numpy as np

# um dado honesto: valores e probabilidades
x = np.arange(1, 7)
p = np.full(6, 1 / 6)
media = np.sum(x * p)                  # E[X]
var = np.sum((x - media) ** 2 * p)     # Var(X)
print(f"dado: E[X] = {media:.4f}   Var(X) = {var:.4f}")
print(f"E[X^2] = {np.sum(x**2 * p):.4f}"
      f"   E[X]^2 = {media**2:.4f}")

# aposta: cara multiplica o capital por 1,5; coroa, por 0,5
fatores = {"cara": 1.5, "coroa": 0.5}
for n in (1, 2):
    W = [float(np.prod([fatores[c] for c in seq]))
         for seq in product(fatores, repeat=n)]
    print(f"{n} rodada(s): resultados {sorted(W)}"
          f"  E[W] = {np.mean(W):.2f}")
