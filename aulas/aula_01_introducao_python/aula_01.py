"""
Aula 1 — Introdução ao Python para Finanças
Núcleo Quant | Liga de Mercado Financeiro

Tópicos:
  1. NumPy: arrays e operações vetorizadas
  2. Matplotlib: visualização de séries de preços
  3. Simulação de caminho de preços (Random Walk)
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. NumPy — operações básicas com arrays
# -------------------------------------------------------------------

# Cria um array de retornos diários simulados (%)
np.random.seed(42)
retornos = np.random.normal(loc=0.0005, scale=0.01, size=252)

print("=== Estatísticas dos retornos simulados ===")
print(f"  Média diária  : {retornos.mean():.4%}")
print(f"  Volatilidade  : {retornos.std():.4%}")
print(f"  Retorno mínimo: {retornos.min():.4%}")
print(f"  Retorno máximo: {retornos.max():.4%}")


# -------------------------------------------------------------------
# 2. Cálculo de preços a partir de retornos (compounding)
# -------------------------------------------------------------------

preco_inicial = 100.0
precos = preco_inicial * np.cumprod(1 + retornos)

print(f"\n  Preço inicial : R$ {preco_inicial:.2f}")
print(f"  Preço final   : R$ {precos[-1]:.2f}")
print(f"  Retorno total : {(precos[-1] / preco_inicial - 1):.2%}")


# -------------------------------------------------------------------
# 3. Simulação de múltiplos caminhos de preço (Monte Carlo simples)
# -------------------------------------------------------------------

n_simulacoes = 500
n_dias = 252
mu = 0.0005    # drift diário
sigma = 0.01   # vol diária

simulacoes = np.zeros((n_dias, n_simulacoes))
simulacoes[0] = preco_inicial

for t in range(1, n_dias):
    choques = np.random.normal(mu, sigma, n_simulacoes)
    simulacoes[t] = simulacoes[t - 1] * (1 + choques)


# -------------------------------------------------------------------
# 4. Visualizações
# -------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Aula 1 — Introdução ao Python para Finanças", fontsize=13)

# Gráfico 1: caminho único de preços
axes[0].plot(precos, color="steelblue", linewidth=1.5)
axes[0].axhline(preco_inicial, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_title("Série de Preços Simulada (1 caminho)")
axes[0].set_xlabel("Dias")
axes[0].set_ylabel("Preço (R$)")

# Gráfico 2: Monte Carlo
axes[1].plot(simulacoes, alpha=0.05, color="steelblue")
axes[1].plot(simulacoes.mean(axis=1), color="red", linewidth=1.5, label="Média")
axes[1].axhline(preco_inicial, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_title(f"Simulação Monte Carlo ({n_simulacoes} caminhos)")
axes[1].set_xlabel("Dias")
axes[1].set_ylabel("Preço (R$)")
axes[1].legend()

plt.tight_layout()
plt.savefig("aula_01_plot.png", dpi=120)
plt.show()
print("\nGráfico salvo em aula_01_plot.png")
