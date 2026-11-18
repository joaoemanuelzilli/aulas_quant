import numpy as np
from scipy.stats import norm

# log-retorno de um ano: média e desvio-padrão (hipótese)
mu, sigma = 0.10, 0.48
x = np.log(0.70)                 # preço cai 30% <=> r < ln 0,70
z = (x - mu) / sigma             # padronizar
print(f"ln 0,70 = {x:.4f}   z = {z:.3f}   P = {norm.cdf(z):.4f}")
print(f"sem padronizar: {norm.cdf(x, loc=mu, scale=sigma):.4f}")
print(f"quantil de 5% da normal padrão: {norm.ppf(0.05):.4f}")

# lognormal: o preço daqui a um ano, partindo de R$ 30
P0 = 30.0
mediana = P0 * np.exp(mu)
media = P0 * np.exp(mu + sigma**2 / 2)
z_media = (np.log(media / P0) - mu) / sigma
print(f"mediana R$ {mediana:.2f}   média R$ {media:.2f}")
print(f"P(preço acima da média) = {norm.sf(z_media):.3f}")
