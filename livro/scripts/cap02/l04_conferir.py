import sys

import pandas as pd

sys.path.insert(0, "..")                 # livro/scripts/
from conferir_fonte import comparar, conferir, ler_fontes

b3, yahoo, volume, eventos = ler_fontes()

# um papel: PETR4 não teve desdobramento, não há cortes
t = comparar(b3["PETR4"], yahoo["PETR4.SA"],
             volume["PETR4.SA"], cortes=[])
colunas = ["b3", "yahoo", "volume", "ret_b3", "ret_yahoo", "dif"]
print(t.loc["2011-08-12":"2011-08-18", colunas].round(4))

# os oito papéis, de 2010 a 2025
papeis = ["ABEV3", "BBAS3", "BOVA11", "ITUB4", "MGLU3",
          "PETR4", "VALE3", "WEGE3"]
tudo, div = conferir(papeis)
n = tudo["dif"].notna().sum()
print(f"\nretornos comparados: {n}")
print(f"divergências acima de 0,5 ponto: {len(div)} "
      f"({len(div) / n:.1%}), {(div.index.year <= 2015).sum()} "
      f"delas até 2015")
print(div["tipo"].value_counts().to_string())
n_papel = div["papel"].value_counts()
print(f"por papel: de {n_papel.min()} ({n_papel.idxmin()}) a "
      f"{n_papel.max()} ({n_papel.idxmax()})")

casos = [("ITUB4", "2013-05-20", "2013-05-21"),
         ("PETR4", "2014-04-02", "2014-04-03"),
         ("PETR4", "2024-03-08", "2024-03-11")]
colunas = ["papel", "oficial", "yahoo", "ret_b3", "ret_yahoo",
           "tipo"]
exemplos = pd.concat(div[div["papel"] == papel].loc[ini:fim]
                     for papel, ini, fim in casos)
print("\n", exemplos[colunas].round(4), sep="")
