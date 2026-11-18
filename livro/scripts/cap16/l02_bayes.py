import numpy as np

rng = np.random.default_rng(2026)
N = 2_000_000                        # repetições do experimento
viciada = rng.random(N) < 1 / 1000   # sorteia uma das mil
# caras em 10 lançamentos: 10 na viciada, binomial na honesta
k = np.where(viciada, 10, rng.binomial(10, 0.5, N))
dez = k == 10                        # o que você observou
p = viciada[dez].mean()              # P(viciada | 10 caras)
ep = np.sqrt(p * (1 - p) / dez.sum())
exato = 1024 / (1024 + 999)
print(f"experimentos com 10 caras: {dez.sum()}")
print(f"simulado: {p:.4f}  (erro-padrão {ep:.4f})")
print(f"exato:    {exato:.4f}")
assert abs(p - exato) < 4 * ep
