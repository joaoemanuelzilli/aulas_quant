import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = yf.download('PETR4.SA', start='2023-01-01', end='2024-01-15')

# Corrigir MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

prices = data['Close']
log_returns = np.log(prices / prices.shift(1)).dropna()

train_size = int(len(log_returns) * 0.8)

train_logs = log_returns[:train_size]
test_logs = log_returns[train_size:]
test_prices = prices[train_size + 1:]

# ==============================
# 4. Walk-forward ARIMA
# ==============================
history = list(train_logs)
predictions_logs = []
predicted_prices = []

last_price = prices.iloc[train_size]

for t in range(len(test_logs)):
    
    model = ARIMA(history, order=(1, 0, 1))
    model_fit = model.fit()
    
    # previsão do log-retorno
    yhat = model_fit.forecast()[0]
    predictions_logs.append(yhat)
    
    last_price = last_price * np.exp(yhat)
    predicted_prices.append(last_price)
    
    # atualiza histórico com valor real
    history.append(test_logs.iloc[t])

rmse = np.sqrt(np.mean((test_prices.values - np.array(predicted_prices)) ** 2))
print(f'RMSE do modelo ARIMA: {rmse:.2f}')

    

plt.figure(figsize=(12, 6))
plt.plot(test_prices.index, test_prices, label='Preço Real')
plt.plot(test_prices.index, predicted_prices, linestyle='--', label='Previsão ARIMA')
plt.title('Modelo ARIMA da PETRA')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.grid()
plt.show()


print("\nÚltimos valores reais:")
print(test_prices.tail())

print("\nÚltimos valores previstos:")
print(pd.Series(predicted_prices, index=test_prices.index).tail())