import numpy as np
from carteira import perda

L = perda["2015":"2024"]
for a in (0.95, 0.99):
    print(f"VaR histórico {a:.0%}: "
          f"R$ {np.quantile(L, a):,.0f}")

# a resposta depende da janela
for ini, fim in (("2015", "2019"), ("2020", "2024"),
                 ("2024", "2024")):
    print(f"VaR histórico 99%, {ini} a {fim}: "
          f"R$ {np.quantile(perda[ini:fim], 0.99):,.0f}")
