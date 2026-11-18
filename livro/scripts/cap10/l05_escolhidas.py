"""As ações do conjunto de dados do livro contra o Ibovespa."""
import pandas as pd

px = pd.read_csv("../../dados/precos_ajustados.csv",
                 index_col=0, parse_dates=True)
acoes = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
         "ABEV3.SA", "BBAS3.SA"]
p = px[acoes + ["^BVSP"]]
inicio = p.loc[:"2010"].ffill().iloc[-1]   # último preço de 2010
fim = p.loc[:"2025"].ffill().iloc[-1]      # último preço de 2025
total = fim / inicio
pregoes = px["PETR4.SA"].loc["2011":"2025"].count()
anos = pregoes / 252                       # convenção de 252 dias
anual = (total ** (1 / anos) - 1).map("{:.1%}".format)
anual.index = anual.index.str.replace(".SA", "")
print(anual.to_frame("a.a.").T.to_string())
carteira = total[acoes].mean()     # pesos iguais, sem rebalancear
print(f"carteira das seis: {carteira ** (1 / anos) - 1:.1%} a.a.")
