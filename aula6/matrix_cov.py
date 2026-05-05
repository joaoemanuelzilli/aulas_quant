import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

ativos = ["PETR4.SA", "VALE3.SA", "AAPL", "MSFT", "USDBRL=X"]
data = yf.download(ativos, start="2022-01-01", end="2025-01-01")["Close"]


# Converter ações americanas para BRL, importante ao se trabalhar com moedas diferentes
data["AAPL_BRL"] = data["AAPL"] * data["USDBRL=X"]
data["MSFT_BRL"] = data["MSFT"] * data["USDBRL=X"]

# Manter apenas ativos na mesma moeda
data = data[["PETR4.SA", "VALE3.SA", "AAPL_BRL", "MSFT_BRL"]]


retornos= np.log(data/data.shift(1))
retornos= retornos.dropna()
cov = retornos.cov() * 252

plt.figure(figsize=(10, 8))
sns.set_theme(style="white")

heatmap = sns.heatmap(
    cov, 
    annot= True,          
    cmap="RdYlGn",        
    linewidths=0.8,        
    cbar_kws={"label": "Nível de Covariância"}
)




plt.title("Matriz de Covariância (2022-2025)", fontsize=16, pad=20)
plt.xticks(rotation=0)
plt.show()