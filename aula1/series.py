import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

# período
inicio = "2018-01-01"
fim = "2023-01-01"

# índices
indices = ["^DJI", "^BVSP"]

# baixar dados
df = yf.download(indices, start=inicio, end=fim)

# preços de fechamento
close = df["Close"]

# calcular retornos
retornos = close.pct_change().dropna()

# variável independente (Dow Jones)
X = retornos["^DJI"].values.reshape(-1,1)

# variável dependente (Ibovespa)
y = retornos["^BVSP"].values

# regressão linear
modelo = LinearRegression()
modelo.fit(X,y)

# previsão
y_pred = modelo.predict(X)

# gráfico
plt.figure(figsize=(8,6))
plt.scatter(X,y,alpha=0.5)
plt.plot(X,y_pred,color="red")

plt.xlabel("Retorno Dow Jones")
plt.ylabel("Retorno Ibovespa")
plt.title("Regressão Linear: Dow Jones vs Ibovespa")

#salva o gráfico em outputs/retornos.png
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "series_regressao.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Gráfico salvo em: {output_file}")

# coeficientes
print("Intercepto:", modelo.intercept_)
print("Coeficiente:", modelo.coef_[0])