import pandas as pd

ohlcv = pd.read_csv("../../dados/bruto/petr4_ohlcv.csv",
                    index_col=0, parse_dates=True)
ohlcv = ohlcv[["Open", "High", "Low", "Close", "Volume"]]
print(ohlcv.loc["2024-12-18":"2024-12-19"].round(2))

# testes de coerência: a máxima é o maior preço do dia,
# a mínima é o menor, e um pregão de verdade tem negócios
corpo = ohlcv[["Open", "Close"]]
fora = ((ohlcv["High"] < corpo.max(axis=1) - 0.005)
        | (ohlcv["Low"] > corpo.min(axis=1) + 0.005))
parado = ohlcv["Volume"] == 0

print(f"\nlinhas: {len(ohlcv)}   incoerentes: {fora.sum()}   "
      f"volume zero: {parado.sum()}")
print(ohlcv.loc["2011-08-15":"2011-08-17",
                ["Close", "Volume"]].round(3))
