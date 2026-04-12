import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

ticker = "PETR4.SA"
df = yf.download('PETR4.SA', start='2023-01-01', end='2023-01-15')


df['Log_Ret'] = 100 * np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# O modelo é treinado sobre os retornos logarítmicos
model = arch_model(df['Log_Ret'], vol='Garch', p=1, q=1, dist='Normal')
res = model.fit(disp='off')

# 4. Visualização dos Resultados
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot dos Retornos (em azul claro no fundo)
ax1.plot(df.index, df['Log_Ret'], color='royalblue', alpha=0.3, label='Retornos Logarítmicos (%)')
ax1.set_ylabel('Retorno Percentual (%)')
ax1.set_xlabel('Data')

# Plot da Volatilidade Condicional (em vermelho)
ax2 = ax1.twinx()
ax2.plot(df.index, res.conditional_volatility, color='red', lw=1.5, label='Volatilidade GARCH(1,1)')
ax2.set_ylabel('Volatilidade Condicional')

plt.title(f'Petrobras ({ticker}): Retornos Log vs. Volatilidade Estimada')
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
plt.grid(alpha=0.2)
plt.show()

# 5. Exibição dos dados recentes e sumário
print(df[['Close', 'Log_Ret']].tail())
print(res.summary())
