"""
Aula 4 — Otimização de Portfólio (Fronteira Eficiente de Markowitz)
Núcleo Quant | Liga de Mercado Financeiro

Tópicos:
  1. Retorno e risco esperados de um portfólio
  2. Simulação de Monte Carlo de portfólios aleatórios
  3. Portfólio de máximo Índice de Sharpe
  4. Portfólio de mínima variância
  5. Fronteira Eficiente via cvxpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cp


# -------------------------------------------------------------------
# 1. Download de dados
# -------------------------------------------------------------------

TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA"]
RF_ANUAL = 0.1075  # taxa livre de risco (Selic aproximada)
RF_DIARIA = RF_ANUAL / 252

print("Baixando dados ...")
dados = yf.download(TICKERS, start="2020-01-01", end="2024-12-31",
                    auto_adjust=True)["Close"].dropna()

retornos = dados.pct_change().dropna()
mu = retornos.mean() * 252          # retornos anualizados
cov = retornos.cov() * 252          # covariância anualizada
n = len(TICKERS)

print(f"\nAtivos: {TICKERS}")
print(f"\nRetornos esperados anuais:\n{mu.round(4)}")


# -------------------------------------------------------------------
# 2. Simulação de Monte Carlo (portfólios aleatórios)
# -------------------------------------------------------------------

N_SIM = 5000
port_ret = np.zeros(N_SIM)
port_vol = np.zeros(N_SIM)
port_sharpe = np.zeros(N_SIM)
port_weights = np.zeros((N_SIM, n))

for i in range(N_SIM):
    w = np.random.dirichlet(np.ones(n))
    r = w @ mu.values
    v = np.sqrt(w @ cov.values @ w)
    port_ret[i] = r
    port_vol[i] = v
    port_sharpe[i] = (r - RF_ANUAL) / v
    port_weights[i] = w

idx_sharpe = np.argmax(port_sharpe)
idx_minvol = np.argmin(port_vol)

print(f"\n=== Portfólio Máx. Sharpe (Monte Carlo) ===")
for ticker, weight in zip(TICKERS, port_weights[idx_sharpe]):
    print(f"  {ticker}: {weight:.2%}")
print(f"  Retorno : {port_ret[idx_sharpe]:.2%}")
print(f"  Vol.    : {port_vol[idx_sharpe]:.2%}")
print(f"  Sharpe  : {port_sharpe[idx_sharpe]:.4f}")


# -------------------------------------------------------------------
# 3. Fronteira Eficiente via cvxpy
# -------------------------------------------------------------------

alvo_retornos = np.linspace(mu.min(), mu.max(), 60)
ef_vol = []
ef_ret = []

for alvo in alvo_retornos:
    w = cp.Variable(n)
    variancia = cp.quad_form(w, cov.values)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w @ mu.values == alvo,
    ]
    prob = cp.Problem(cp.Minimize(variancia), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
        ef_vol.append(np.sqrt(variancia.value))
        ef_ret.append(alvo)


# -------------------------------------------------------------------
# 4. Visualização
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Aula 4 — Otimização de Portfólio (Markowitz)", fontsize=13)

sc = ax.scatter(port_vol, port_ret, c=port_sharpe, cmap="viridis",
                s=5, alpha=0.5, label="Portfólios Aleatórios")
plt.colorbar(sc, ax=ax, label="Índice de Sharpe")

if ef_vol:
    ax.plot(ef_vol, ef_ret, color="red", linewidth=2, label="Fronteira Eficiente")

ax.scatter(port_vol[idx_sharpe], port_ret[idx_sharpe], marker="*",
           color="gold", s=300, zorder=5, label="Máx. Sharpe")
ax.scatter(port_vol[idx_minvol], port_ret[idx_minvol], marker="D",
           color="cyan", s=100, zorder=5, label="Mín. Variância")

ax.set_xlabel("Volatilidade Anual")
ax.set_ylabel("Retorno Anual Esperado")
ax.legend()
plt.tight_layout()
plt.savefig("aula_04_plot.png", dpi=120)
plt.show()
print("\nGráfico salvo em aula_04_plot.png")
