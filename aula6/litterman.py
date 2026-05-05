import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

tickers = ["AAPL", "PETR4.SA", "VALE3.SA"]
raw_data = yf.download(tickers, period="3y")

if 'Adj Close' in raw_data.columns:
    dados = raw_data['Adj Close']
else:
    dados = raw_data['Close']

retornos = dados.pct_change().dropna()

sigma = retornos.cov().values * 252
w_mkt = np.array([0.40, 0.30, 0.30]) 
lmbda = 3.0   
tau = 0.05    

pi = lmbda * sigma @ w_mkt 

# opinião 1: PETR4 terá retorno de 15%
# opinião 2: AAPL superará VALE3 em 5%


Q = np.array([0.20, 0.05])
P = np.array([
    [0, 1, 0],  # PETR4 absoluta (segunda coluna)
    [1, 0, -1]  # AAPL (1ª) vs VALE3 (3ª)
])

omega = np.diag(np.diag(P @ (tau * sigma) @ P.T))
inv_tau_sigma = np.linalg.inv(tau * sigma)
inv_omega = np.linalg.inv(omega)

A = inv_tau_sigma + P.T @ inv_omega @ P
b = inv_tau_sigma @ pi + P.T @ inv_omega @ Q
er_bl = np.linalg.solve(A, b)

w_bl = np.linalg.solve(lmbda * sigma, er_bl)
w_bl = np.maximum(w_bl, 0) 
w_bl /= np.sum(w_bl)      # Re-balanceamento

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
labels = sorted(tickers)
x = np.arange(len(labels))
ax1.bar(x - 0.2, w_mkt, width=0.4, label='Equilíbrio (Mercado)', color='#95a5a6', alpha=0.6)
ax1.bar(x + 0.2, w_bl, width=0.4, label='Black-Litterman', color='#2980b9')
ax1.set_ylabel('Peso (%)')
ax1.set_title('Alocação de Ativos: Impacto das Opiniões', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()


vol_mkt = np.sqrt(w_mkt.T @ sigma @ w_mkt)
ret_mkt = w_mkt.T @ pi

vol_bl = np.sqrt(w_bl.T @ sigma @ w_bl)
ret_bl = w_bl.T @ er_bl

ax2.scatter(vol_mkt, ret_mkt, color='gray', s=200, label='Portfólio Mercado', zorder=5)
ax2.scatter(vol_bl, ret_bl, color='blue', s=200, label='Portfólio BL', zorder=5)

for i, ticker in enumerate(labels):
    ax2.annotate(ticker, (np.sqrt(sigma[i,i]), er_bl[i]), xytext=(5,5), textcoords='offset points')
    ax2.scatter(np.sqrt(sigma[i,i]), er_bl[i], alpha=0.5)

ax2.set_xlabel('Risco (Volatilidade Anualizada)')
ax2.set_ylabel('Retorno Esperado Anualizado')
ax2.set_title('Espaço Risco-Retorno', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.show()