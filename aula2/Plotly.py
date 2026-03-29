import yfinance as yf
import pandas as pd
import plotly.express as px

# =========================
# Baixando dados
# =========================
ticker = yf.Ticker("PETR4.SA")  # Petrobras
dados = ticker.history(period="1y")

print(dados.head())

# fig = px.line(dados, x=dados.index, y=['Close'], title='Preço Interativo')
# fig.show()

# =========================
# 📉 RETORNO
# =========================

dados['Retorno'] = dados['Close'].pct_change()

# fig = px.bar(dados, x=dados.index, y='Retorno',
#               title='Retorno Diário')
# fig.show()

# =========================
# 📊 HISTOGRAMA (RISCO)
# =========================

# fig = px.histogram(dados.dropna(), x='Retorno',
#                    nbins=30,
#                    title='Distribuição de Retornos')
# fig.show()

# =========================
# 📊 Comparação (Preco)
# =========================

# ticker_petr = yf.Ticker("PETR4.SA")  # Petrobras
# ticket_ibov = yf.Ticker("^BVSP")  # Ibovespa

# petr = ticker_petr.history(period="1y")['Close'].rename('PETR4')
# ibov = (ticket_ibov.history(period="1y")['Close'].rename('IBOVESPA'))/10000

# dados = pd.concat([petr, ibov], axis=1)

# # Plotar comparação
# fig = px.line(dados, x=dados.index, y=['PETR4', 'IBOVESPA'],
#               title='Comparação: PETR4 vs Ibovespa',
#               labels={'value': 'Preço', 'index': 'Data'})
# fig.show()