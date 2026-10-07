import numpy as np

mu_a, mu_b = 0.10, 0.15          # retornos esperados
s_a, s_b, rho = 0.20, 0.30, 0.30  # volatilidades e correlação


def carteira(w):
    """Retorno e volatilidade com peso w em A e 1 - w em B."""
    mu = w * mu_a + (1 - w) * mu_b
    var = ((w * s_a) ** 2 + ((1 - w) * s_b) ** 2
           + 2 * w * (1 - w) * rho * s_a * s_b)
    return mu, np.sqrt(var)


mu, vol = carteira(0.5)
print(f"50/50:  retorno {mu:.2%}  volatilidade {vol:.2%}")
print(f"média das volatilidades: {(s_a + s_b) / 2:.2%}")

cov = rho * s_a * s_b
w_min = (s_b**2 - cov) / (s_a**2 + s_b**2 - 2 * cov)
mu, vol = carteira(w_min)
print(f"mínima variância: w_A = {w_min:.4f}  "
      f"retorno {mu:.2%}  volatilidade {vol:.2%}")
