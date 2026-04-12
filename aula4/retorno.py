import yfinance as yf
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Baixar dados
df = yf.download("VALE3.SA", start="2020-01-01")

# Série de preço
preco = df['Close'].dropna()

# Série de retorno
log_returns = np.log(preco / preco.shift(1)).dropna()

# Função ADF
def adf_test(series, name):
    result = adfuller(series)
    print(f"\nTeste ADF - {name}")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] < 0.05:
        print(f"A serie {name} e estacionaria.")       
    else:     print(f"A serie {name} nao e estacionaria.")

# Testando
adf_test(preco, "Preco")
adf_test(log_returns, "Retorno")