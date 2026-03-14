"""
Aula 3 — Estatísticas para Finanças
Núcleo Quant | Liga de Mercado Financeiro

Tópicos:
  1. Distribuição normal vs. retornos reais (fat tails)
  2. Assimetria (skewness) e curtose (kurtosis)
  3. Teste de normalidade (Jarque-Bera)
  4. Regressão linear simples (CAPM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf


# -------------------------------------------------------------------
# 1. Download de dados
# -------------------------------------------------------------------

print("Baixando dados ...")
dados = yf.download(["PETR4.SA", "^BVSP"], start="2020-01-01", end="2024-12-31",
                    auto_adjust=True)["Close"].dropna()

retornos = dados.pct_change().dropna()
r_ativo = retornos["PETR4.SA"]
r_mercado = retornos["^BVSP"]


# -------------------------------------------------------------------
# 2. Momentos estatísticos
# -------------------------------------------------------------------

print("\n=== Momentos dos Retornos — PETR4.SA ===")
print(f"  Média       : {r_ativo.mean():.4%}")
print(f"  Volatilidade: {r_ativo.std():.4%}")
print(f"  Assimetria  : {r_ativo.skew():.4f}")
print(f"  Curtose     : {r_ativo.kurtosis():.4f}")


# -------------------------------------------------------------------
# 3. Teste de normalidade (Jarque-Bera)
# -------------------------------------------------------------------

jb_stat, jb_pvalue = stats.jarque_bera(r_ativo)
print(f"\n=== Teste Jarque-Bera ===")
print(f"  Estatística : {jb_stat:.2f}")
print(f"  p-valor     : {jb_pvalue:.2e}")
if jb_pvalue < 0.05:
    print("  Rejeita normalidade ao nível de 5%.")
else:
    print("  Não rejeita normalidade ao nível de 5%.")


# -------------------------------------------------------------------
# 4. Regressão linear — CAPM (PETR4 x Ibovespa)
# -------------------------------------------------------------------

slope, intercept, r_value, p_value, std_err = stats.linregress(r_mercado, r_ativo)
beta = slope
alpha = intercept

print(f"\n=== Regressão CAPM — PETR4 vs. Ibovespa ===")
print(f"  Alpha (α)   : {alpha:.4%}")
print(f"  Beta (β)    : {beta:.4f}")
print(f"  R²          : {r_value**2:.4f}")
print(f"  p-valor (β) : {p_value:.2e}")


# -------------------------------------------------------------------
# 5. Visualizações
# -------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Aula 3 — Estatísticas para Finanças", fontsize=13)

# Histograma vs. distribuição normal
x = np.linspace(r_ativo.min(), r_ativo.max(), 200)
pdf_normal = stats.norm.pdf(x, r_ativo.mean(), r_ativo.std())

axes[0].hist(r_ativo, bins=80, density=True, color="steelblue",
             edgecolor="white", alpha=0.7, label="Retornos Reais")
axes[0].plot(x, pdf_normal, color="red", linewidth=2, label="Normal Teórica")
axes[0].set_title("Distribuição dos Retornos vs. Normal")
axes[0].set_xlabel("Retorno Diário")
axes[0].legend()

# Q-Q plot
stats.probplot(r_ativo, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot — PETR4")

# Scatter CAPM
axes[2].scatter(r_mercado, r_ativo, alpha=0.2, s=10, color="steelblue")
x_line = np.array([r_mercado.min(), r_mercado.max()])
axes[2].plot(x_line, intercept + slope * x_line, color="red", linewidth=1.5,
             label=f"β = {beta:.2f}  R² = {r_value**2:.2f}")
axes[2].set_title("Regressão CAPM — PETR4 vs. Ibovespa")
axes[2].set_xlabel("Retorno Ibovespa")
axes[2].set_ylabel("Retorno PETR4")
axes[2].legend()

plt.tight_layout()
plt.savefig("aula_03_plot.png", dpi=120)
plt.show()
print("\nGráfico salvo em aula_03_plot.png")
