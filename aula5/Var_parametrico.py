import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt

# Dados
df = yf.download("VALE3.SA", start="2020-01-01")
price = df['Close'].squeeze()
returns = np.log(price / price.shift(1)).dropna()

# Parâmetros
confianca = 0.95
alpha = 1 - confianca

media = returns.mean()
sigma = returns.std()

z = norm.ppf(alpha)
var_parametrico = media + z * sigma

print(f"VaR Parametrico (95%): {var_parametrico:.4f}")

T = 100  # horizonte de 100 dias
var_dias = media * T + z * sigma * np.sqrt(T)
print(f"VaR Parametrico (100 dias): {var_dias:.4f}")

# Plot
x = np.linspace(returns.min(), returns.max(), 1000)
pdf = norm.pdf(x, media, sigma)

plt.plot(x, pdf, label="Distribuição Normal")
plt.axvline(var_parametrico, linestyle='--', linewidth=2, label="VaR")
plt.title("VaR Paramétrico")
plt.legend()
plt.show()