"""Contas de apoio: poder do teste, acaso e liquidez de MGLU3."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from projeto import DADOS, TREINO, alvo_momentum, carregar

# 1. chance de t >= 2 quando a razão de informação é 0,5
for anos in (6, 16, 32):
    chance = norm.cdf(0.5 * np.sqrt(anos) - 2)
    print(f"RI verdadeira 0,5, {anos:2d} anos: "
          f"P(t >= 2) = {chance:.0%}")
m = 0.5 * np.sqrt(6)                # t esperado em seis anos
print(f"em 6 anos: confirmada {1 - norm.cdf(2 - m):.0%}, "
      f"refutada (t <= 0) {norm.cdf(-m):.0%}, inconclusiva "
      f"{norm.cdf(2 - m) - norm.cdf(-m):.0%}")
print(f"anos para 80% de chance: "
      f"{((2 + norm.ppf(0.8)) / 0.5) ** 2:.0f}")

# 2. melhor de N variantes sem habilidade, em 7 anos de treino
g = 0.5772
for n in (4, 12):
    z = ((1 - g) * norm.ppf(1 - 1 / n)
         + g * norm.ppf(1 - 1 / (n * np.e)))
    print(f"máximo esperado de {n:2d} tentativas: "
          f"{z / np.sqrt(7):.2f}")

# 3. liquidez de MGLU3 quando a regra a compra, no treino
ler = lambda arq: pd.read_csv(DADOS / arq, index_col=0,
                              parse_dates=True)
fin = ler("volume_financeiro.csv")     # R$ por dia, oficial
fin = fin[["MGLU3.SA", "PETR4.SA"]].loc["2013":"2017"]
print((fin.groupby(fin.index.year).median() / 1e6).round(1)
      .T.to_string())
px, ret, cdi = carregar()
alvo = alvo_momentum(px, 12, 2).loc["2015-12":TREINO[1]]
dentro = alvo["MGLU3.SA"] > 0
print(f"MGLU3 entra na carteira com o sinal de "
      f"{dentro.idxmax().date()}; meses dentro até 2019: "
      f"{dentro.sum()} de {len(dentro)}")
vol = ret.loc["2016", "MGLU3.SA"].std()
v = fin.loc["2016", "MGLU3.SA"].median()
print(f"volatilidade diária em 2016: {vol:.1%}")
for capital in (1e5, 1e6, 1e7):
    q = capital / 2                      # metade da carteira
    impacto = 2 / 3 * vol * np.sqrt(q / v)
    print(f"carteira de R$ {capital:>10,.0f}: ordem = "
          f"{q / v:5.1%} do volume, impacto "
          f"{impacto * 1e4:4.0f} pb")
