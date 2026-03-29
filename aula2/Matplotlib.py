import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Baixando dados
# =========================

ticker = yf.Ticker("PETR4.SA")  # Petrobras
dados = ticker.history(period="1y")

print(dados.head())
dados = dados.dropna()
# =========================
# 📈 MATPLOTLIB
# =========================

plt.figure(figsize=(10,6))
# plt.bar(dados.index, dados['Close'])
plt.plot(dados.index, dados['Close'])
plt.title('Preço da ação - Petr4')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.grid()
plt.show()

# =========================
# 📉 RETORNO
# =========================

dados['Retorno'] = dados['Close'].pct_change()

plt.figure(figsize=(10,5))
plt.plot(dados.index, dados['Retorno'])
plt.title('Retorno Diário')
plt.grid()
plt.show()

# =========================
# 📊 HISTOGRAMA (RISCO)
# =========================

plt.figure(figsize=(8,5))
plt.hist(dados['Retorno'].dropna(), bins=30)
plt.title('Distribuição dos Retornos')
plt.show()
