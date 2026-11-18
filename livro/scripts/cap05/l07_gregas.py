import numpy as np

from calculo import bs, gregas

p = dict(S=30.0, K=30.0, T=63 / 252, r=0.14, sigma=0.30)
z = np.random.default_rng(42).standard_normal(1_000_000)


def mc(S, K, T, r, sigma):
    """Payoff descontado de cada cenário (mesmos z sempre)."""
    ST = S * np.exp((r - sigma**2 / 2) * T
                    + sigma * np.sqrt(T) * z)
    return np.exp(-r * T) * np.maximum(ST - K, 0)


def dif(f, nome, h, ordem=1):
    """Diferença central de f no parâmetro `nome`."""
    sobe = f(**{**p, nome: p[nome] + h})
    desce = f(**{**p, nome: p[nome] - h})
    if ordem == 2:
        return (sobe - 2 * f(**p) + desce) / h**2
    return (sobe - desce) / (2 * h)


casos = [("delta", "S", 0.3, 1, 1), ("gama", "S", 0.3, 2, 1),
         ("vega", "sigma", 0.01, 1, 1),
         ("theta", "T", 0.01, 1, -1),
         ("rho", "r", 0.01, 1, 1)]
fechada = gregas(**p)
print("grega   fórmula   dif. finita   Monte Carlo (e.p.)")
for g, nome, h, ordem, sinal in casos:
    df = sinal * dif(bs, nome, h, ordem)
    cen = sinal * dif(mc, nome, h, ordem)   # um valor por cenário
    ep = cen.std(ddof=1) / np.sqrt(len(z))
    print(f"{g:<6} {fechada[g]:9.4f} {df:11.4f}"
          f" {cen.mean():11.4f} ({ep:.4f})")
