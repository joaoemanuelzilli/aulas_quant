"""
Aula 5 — Gestão de Risco: VaR e CVaR
Núcleo Quant | Liga de Mercado Financeiro

Tópicos:
  1. Value at Risk (VaR) — método histórico
  2. VaR paramétrico (distribuição normal)
  3. Conditional VaR (CVaR / Expected Shortfall)
  4. Backtesting do VaR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf


# -------------------------------------------------------------------
# Parâmetros globais
# -------------------------------------------------------------------

TICKER = "PETR4.SA"
NIVEL_CONFIANCA = 0.95   # 95%
CAPITAL = 1_000_000      # R$ 1 milhão
JANELA_BACKTEST = 252    # dias de lookback para VaR histórico


def var_historico(retornos: pd.Series, nivel: float = 0.95) -> float:
    """VaR histórico pelo quantil empírico."""
    return float(-retornos.quantile(1 - nivel))


def var_parametrico(retornos: pd.Series, nivel: float = 0.95) -> float:
    """VaR paramétrico assumindo distribuição normal."""
    mu = retornos.mean()
    sigma = retornos.std()
    z = stats.norm.ppf(1 - nivel)
    return float(-(mu + z * sigma))


def cvar(retornos: pd.Series, nivel: float = 0.95) -> float:
    """CVaR (Expected Shortfall) histórico."""
    threshold = -var_historico(retornos, nivel)
    tail = retornos[retornos <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


# -------------------------------------------------------------------
# 1. Download de dados
# -------------------------------------------------------------------

print("Baixando dados ...")
dados = yf.download(TICKER, start="2018-01-01", end="2024-12-31",
                    auto_adjust=True)["Close"].dropna()

retornos = dados.pct_change().dropna()
print(f"Total de observações: {len(retornos)}")


# -------------------------------------------------------------------
# 2. Cálculo do VaR e CVaR para todo o período
# -------------------------------------------------------------------

var_hist = var_historico(retornos, NIVEL_CONFIANCA)
var_param = var_parametrico(retornos, NIVEL_CONFIANCA)
cvar_val = cvar(retornos, NIVEL_CONFIANCA)

print(f"\n=== Métricas de Risco — {TICKER} (nível {NIVEL_CONFIANCA:.0%}) ===")
print(f"  VaR Histórico      : {var_hist:.4%}  →  R$ {var_hist * CAPITAL:,.0f}")
print(f"  VaR Paramétrico    : {var_param:.4%}  →  R$ {var_param * CAPITAL:,.0f}")
print(f"  CVaR (ES) Histórico: {cvar_val:.4%}  →  R$ {cvar_val * CAPITAL:,.0f}")


# -------------------------------------------------------------------
# 3. Backtesting do VaR histórico (rolling window)
# -------------------------------------------------------------------

var_rolling = pd.Series(index=retornos.index[JANELA_BACKTEST:], dtype=float)
for i in range(JANELA_BACKTEST, len(retornos)):
    janela = retornos.iloc[i - JANELA_BACKTEST:i]
    var_rolling.iloc[i - JANELA_BACKTEST] = var_historico(janela, NIVEL_CONFIANCA)

retornos_teste = retornos.iloc[JANELA_BACKTEST:]
violacoes = retornos_teste < -var_rolling
taxa_violacao = violacoes.sum() / len(violacoes)
taxa_esperada = 1 - NIVEL_CONFIANCA

print(f"\n=== Backtesting VaR ({JANELA_BACKTEST}d rolling) ===")
print(f"  Violações     : {violacoes.sum()} ({taxa_violacao:.2%})")
print(f"  Taxa esperada : {taxa_esperada:.2%}")
resultado = "✓ Adequado" if abs(taxa_violacao - taxa_esperada) < 0.02 else "✗ Revisar modelo"
print(f"  Resultado     : {resultado}")


# -------------------------------------------------------------------
# 4. Visualizações
# -------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f"Aula 5 — Gestão de Risco: VaR e CVaR — {TICKER}", fontsize=13)

# --- Gráfico 1: Distribuição de retornos com VaR/CVaR marcados ---
ax = axes[0]
x = np.linspace(retornos.min(), retornos.max(), 300)
ax.hist(retornos, bins=100, density=True, color="steelblue",
        edgecolor="white", alpha=0.6, label="Retornos")
ax.axvline(-var_hist, color="orange", linewidth=1.5,
           linestyle="--", label=f"VaR Hist. ({NIVEL_CONFIANCA:.0%}) = {var_hist:.2%}")
ax.axvline(-var_param, color="green", linewidth=1.5,
           linestyle="--", label=f"VaR Param. = {var_param:.2%}")
ax.axvline(-cvar_val, color="red", linewidth=1.5,
           linestyle="--", label=f"CVaR = {cvar_val:.2%}")
ax.set_title("Distribuição de Retornos com VaR e CVaR")
ax.set_xlabel("Retorno Diário")
ax.legend()

# --- Gráfico 2: Backtesting ---
ax2 = axes[1]
retornos_teste.plot(ax=ax2, color="steelblue", alpha=0.6, linewidth=0.8,
                   label="Retorno Realizado")
(-var_rolling).plot(ax=ax2, color="orange", linewidth=1.2, label="−VaR Rolling")
ax2.scatter(retornos_teste.index[violacoes],
            retornos_teste[violacoes], color="red", s=15, zorder=5,
            label=f"Violações ({violacoes.sum()})")
ax2.set_title("Backtesting VaR Histórico (Rolling)")
ax2.set_xlabel("Data")
ax2.legend()

plt.tight_layout()
plt.savefig("aula_05_plot.png", dpi=120)
plt.show()
print("\nGráfico salvo em aula_05_plot.png")
