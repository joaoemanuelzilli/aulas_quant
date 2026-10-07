import numpy as np

import metricas as m
from motor import carregar, cruzamento


def erro_padrao(sharpe_anual, anos, dias=252):
    """Erro-padrão do Sharpe anualizado sob retornos iid."""
    s = sharpe_anual / np.sqrt(dias)          # Sharpe por dia
    ep_dia = np.sqrt((1 + s ** 2 / 2) / (dias * anos))
    return ep_dia * np.sqrt(dias)             # volta a anualizar


preco, ret = carregar("PETR4.SA")
bt = cruzamento(preco, ret["cdi"]).loc["2011":"2025"]
rf = ret["cdi"].loc["2011":"2025"]
anos = len(bt) / 252
sr = m.sharpe(bt["retorno"], rf)
ep = erro_padrao(sr, anos)
print(f"Sharpe {sr:.2f}   erro-padrão {ep:.2f}   "
      f"IC 95%: [{sr - 1.96 * ep:.2f}, {sr + 1.96 * ep:.2f}]")

for alvo in [0.5, 1.0, 2.0]:
    minimo = (1.96 / alvo) ** 2 * (1 + alvo ** 2 / 504)
    print(f"Sharpe {alvo:.1f}: {minimo:5.1f} anos para t = 1.96")

# prazo para rejeitar Sharpe nulo em 80% dos históricos
z80 = 0.8416                              # quantil 80% da normal
print(f"Sharpe 0.5, poder de 80%: "
      f"{((1.96 + z80) / 0.5) ** 2:.1f} anos")

# conferência por simulação: 20 mil históricos de 10 anos
rng = np.random.default_rng(7)
sim = rng.normal(0.5 / 252, 1 / np.sqrt(252), (20_000, 2520))
est = sim.mean(axis=1) / sim.std(axis=1, ddof=1)
est = est * np.sqrt(252)
print(f"simulado: {est.std():.3f}   fórmula: "
      f"{erro_padrao(0.5, 10):.3f}")
