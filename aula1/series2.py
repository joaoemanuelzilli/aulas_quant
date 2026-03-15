import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path


inicio = "2019-01-01"

acoes=['GGBR4.SA','BBAS3.SA','ITUB4.SA']


df = yf.download(acoes, start=inicio, end="2023-11-24")

#print(df.head())

df["Close"].plot()

#salva o gráfico em outputs/retornos.png
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "series_precos.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Gráfico salvo em: {output_file}")


#print(df['Close'].tail(5))