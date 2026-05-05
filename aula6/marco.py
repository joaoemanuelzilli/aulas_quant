import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import yfinance as yf

# 1. Busca de dados
tickers = ['USDBRL=X', 'EURBRL=X', 'GBPBRL=X',"AAPL"]
nomes_dict = {'USDBRL=X': 'dolar', 'EURBRL=X': 'euro', 'GBPBRL=X': 'libra','AAPL':'apple'}




# Baixando e garantindo a seleção da coluna 'Close' corretamente
dados_raw = yf.download(tickers, period='1y')['Close']

# Garantir que as colunas sigam a ordem dos nomes que queremos
dados = dados_raw[tickers].rename(columns=nomes_dict)
dados = dados.dropna()

nomes = list(nomes_dict.values())

# --- Cálculos Estatísticos ---
# Trabalhar diretamente com o objeto prec facilita manter os nomes das colunas
prec = dados
ri = prec.pct_change().dropna()

# Importante: manter como Series/DataFrame para evitar desalinhamento de índices
mi = ri.mean() * 252
sigma = ri.cov() * 252

print('\n++++++++++ Matriz de Covariância Anualizada ++++++++++')
print(sigma)

# --- Monte Carlo ---
vet_R, vet_Vol, vet_W = [], [], []

for _ in range(5000):
    w = np.random.random(len(nomes))
    w /= np.sum(w)
    retorno = np.sum(w * mi.values)
    risco = np.sqrt(np.dot(w.T, np.dot(sigma.values, w)))
    
    vet_R.append(retorno)
    vet_Vol.append(risco)
    vet_W.append(w)

# --- Otimização ---
def vol_port(peso):
    return np.sqrt(np.dot(peso.T, np.dot(sigma.values, peso)))

x0 = np.ones(len(nomes)) / len(nomes)
bounds = tuple((0, 1) for _ in nomes)

# Ajuste na faixa de retorno para cobrir apenas o que é logicamente possível
faixa_ret = np.linspace(min(vet_R), max(vet_R), 30)

plot_risk, plot_ret = [], []

for r in faixa_ret:
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x, r=r: np.sum(x * mi.values) - r}
    ]
    res = minimize(vol_port, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success:
        plot_risk.append(res.fun)
        plot_ret.append(r)

# --- Plots ---
plt.figure(figsize=(10, 6))
plt.scatter(vet_Vol, vet_R, c=(np.array(vet_R)/np.array(vet_Vol)), marker='o', alpha=0.3, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')

plt.plot(plot_risk, plot_ret, 'r--', linewidth=3, label='Fronteira Eficiente')

# Mínima Variância
res_min = minimize(vol_port, x0, method='SLSQP', bounds=bounds, 
                   constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}])
ret_min = np.sum(res_min.x * mi.values)
vol_min = res_min.fun
plt.scatter(vol_min, ret_min, marker='*', s=300, color='gold', edgecolors='black', label='Mín. Variância')

plt.xlabel('Volatilidade Anualizada')
plt.ylabel('Retorno Esperado Anualizado')
plt.legend()
plt.show()