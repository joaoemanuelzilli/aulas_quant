import numpy as np

from calculo import bs

S, K, T, r, sigma = 30.0, 30.0, 63 / 252, 0.14, 0.30
C, P = bs(S, K, T, r, sigma), bs(S, K, T, r, sigma, "venda")
print(f"compra: R$ {C:.4f}   venda: R$ {P:.4f}")
print(f"paridade C - P = S - K e^(-rT): {C - P:.4f}"
      f" = {S - K * np.exp(-r * T):.4f}")

# Monte Carlo: preço lognormal com média de crescimento r
rng = np.random.default_rng(42)
z = rng.standard_normal(1_000_000)
ST = S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * z)
pago = np.exp(-r * T) * np.maximum(ST - K, 0)
ep = pago.std(ddof=1) / np.sqrt(len(z))
print(f"Monte Carlo: R$ {pago.mean():.4f}"
      f"  (erro-padrão {ep:.4f})")
