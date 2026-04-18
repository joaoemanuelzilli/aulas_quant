import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Dados
df = yf.download("VALE3.SA", start="2020-01-01")
price = df['Close'].dropna()
returns = np.log(price / price.shift(1)).dropna()

# Parâmetros
confianca = 0.95
alpha = 1 - confianca

media = returns.mean()
sigma = returns.std()

# Simulação Monte Carlo
n_simulacao = 10000
simulated_returns = np.random.normal(media, sigma, n_simulacao)

var_mc = np.percentile(simulated_returns, alpha * 100)

print(f"VaR Monte Carlo (95%): {var_mc:.4f}")

# Plot
plt.hist(simulated_returns, bins=50, alpha=0.7)
plt.axvline(var_mc, linestyle='--', linewidth=2, label="VaR")
plt.title("VaR Monte Carlo")
plt.legend()
plt.show()