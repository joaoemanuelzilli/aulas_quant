import yfinance as yf
import matplotlib.pyplot as plt

# Definir ativos (ex: Apple e Ibovespa)
ativos = ["AAPL", "^BVSP"]

# Período
inicio = "2020-01-01"
fim = "2024-01-01"

# Baixar dados
dados = yf.download(ativos, inicio, fim)

# Preço de fechamento
fechamento = dados["Close"]

print("=== Dados de fechamento ===")
print(fechamento.head())

# -------------------------------
# 1. Tratar dados faltantes
# -------------------------------
fechamento = fechamento.fillna(method="ffill")

# -------------------------------
# 2. Calcular retornos diários
# -------------------------------
retornos = fechamento.pct_change().dropna()  #retorno percentual=(p(t) -p(t-1))/p(t-1)

#print("\n=== Retornos ===")
#print(retornos.head())

# -------------------------------
# 3. Retorno acumulado
# -------------------------------
retorno_acum = (1 + retornos).cumprod()

# -------------------------------
# 4. Plotar preços
# -------------------------------
plt.figure()
fechamento.plot(title="Preços de Fechamento")
plt.show()

# -------------------------------
# 5. Plotar retorno acumulado
# -------------------------------
plt.figure()
retorno_acum.plot(title="Retorno Acumulado")
plt.show()