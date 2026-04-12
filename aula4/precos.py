import yfinance as yf
from statsmodels.tsa.stattools import adfuller

# Baixar dados
df = yf.download("PETR4.SA", start="2020-01-01")

# Série de preço
preco = df['Close'].dropna()

# Função ADF
result = adfuller(preco)
print(f"p-value: {result[1]}")

if result[1] < 0.05:
        print(f"A série  é estacionária.")       
else:     print(f"A série  não é estacionária.")


