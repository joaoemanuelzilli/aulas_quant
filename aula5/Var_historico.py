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

# VaR Histórico
var_historical = np.percentile(returns, alpha * 100)

print(f"VaR Histórico (95%): {var_historical:.4f}")

# Plot
plt.hist(returns, bins=50, alpha=0.7)
plt.axvline(var_historical, linestyle='--', linewidth=2, label="VaR")
plt.title("VaR Histórico")
plt.legend()
plt.show()