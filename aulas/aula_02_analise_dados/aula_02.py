"""
Aula 2 — Análise de Dados Financeiros com Pandas
Núcleo Quant | Liga de Mercado Financeiro

Tópicos:
  1. Download de dados com yfinance
  2. Cálculo de retornos simples e logarítmicos
  3. Estatísticas descritivas
  4. Correlação entre ativos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# -------------------------------------------------------------------
# 1. Download de dados históricos
# -------------------------------------------------------------------

TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA"]
START = "2020-01-01"
END = "2024-12-31"

print(f"Baixando dados de {START} a {END} ...")
dados = yf.download(TICKERS, start=START, end=END, auto_adjust=True)["Close"]
dados.dropna(how="all", inplace=True)
print(f"Shape dos dados: {dados.shape}\n")
print(dados.tail())


# -------------------------------------------------------------------
# 2. Retornos diários
# -------------------------------------------------------------------

retornos_simples = dados.pct_change().dropna()
retornos_log = np.log(dados / dados.shift(1)).dropna()

print("\n=== Retornos Simples — Estatísticas Descritivas ===")
print(retornos_simples.describe().round(4))


# -------------------------------------------------------------------
# 3. Retorno acumulado
# -------------------------------------------------------------------

retorno_acumulado = (1 + retornos_simples).cumprod()


# -------------------------------------------------------------------
# 4. Matriz de correlação
# -------------------------------------------------------------------

corr = retornos_simples.corr()
print("\n=== Matriz de Correlação ===")
print(corr.round(3))


# -------------------------------------------------------------------
# 5. Visualizações
# -------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Aula 2 — Análise de Dados Financeiros", fontsize=13)

# Preços normalizados
(dados / dados.iloc[0] * 100).plot(ax=axes[0, 0])
axes[0, 0].set_title("Preços Normalizados (base 100)")
axes[0, 0].set_xlabel("")

# Retorno acumulado
retorno_acumulado.plot(ax=axes[0, 1])
axes[0, 1].set_title("Retorno Acumulado")
axes[0, 1].set_xlabel("")

# Histograma de retornos — PETR4
axes[1, 0].hist(retornos_simples["PETR4.SA"], bins=60, color="steelblue", edgecolor="white")
axes[1, 0].set_title("Distribuição de Retornos — PETR4")
axes[1, 0].set_xlabel("Retorno Diário")

# Heatmap de correlação
import seaborn as sns
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Matriz de Correlação")

plt.tight_layout()
plt.savefig("aula_02_plot.png", dpi=120)
plt.show()
print("\nGráfico salvo em aula_02_plot.png")
