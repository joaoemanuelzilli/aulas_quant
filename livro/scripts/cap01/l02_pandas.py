import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)

# uma coluna do DataFrame é uma Series
petr = precos["PETR4.SA"].dropna()
print(type(precos.index).__name__, "|", precos.shape)

# fatiar por data
print(petr.loc["2024-01-02":"2024-01-05"].round(2))

# último preço de cada ano
print(petr.resample("YE").last().tail(3).round(2))
