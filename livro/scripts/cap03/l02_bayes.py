import numpy as np
import pandas as pd

def posterior(prior, sens, falso):
    """P(H | S), com prior = P(H), sens = P(S | H)
    e falso = P(S | não H)."""
    total = sens * prior + falso * (1 - prior)  # prob. total
    return sens * prior / total

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()
base = (r < -0.03).mean()          # taxa-base de dias de queda
print(f"{len(r)} pregões, P(queda > 3%) = {base:.4f}")
print(f"P(queda | alarme) = {posterior(base, 0.80, 0.10):.3f}")

for prior in (0.10, 0.01):         # estratégias que funcionam
    pr = posterior(prior, 0.50, 0.05)
    print(f"P(real) = {prior:.2f} -> P(real | passou) = {pr:.3f}")
