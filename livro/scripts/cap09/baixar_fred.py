"""Baixa a taxa da letra do Tesouro americano de 3 meses (FRED).

Rode uma única vez; grava dados/c09_tbill.csv com a série DTB3
(taxa de desconto, % ao ano, mercado secundário), do Federal
Reserve Bank of St. Louis, com corte em 30/09/2026.
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DESTINO = Path(__file__).resolve().parents[2] / "dados" / "c09_tbill.csv"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

if DESTINO.exists():
    print("já existe:", DESTINO.name, "(nada baixado)")
else:
    resposta = requests.get(URL, timeout=120)
    resposta.raise_for_status()
    tb = pd.read_csv(StringIO(resposta.text), index_col=0,
                     parse_dates=True, na_values=".")
    tb = tb.iloc[:, 0].dropna().loc[:"2026-09-30"]
    tb.index.name, tb.name = "data", "tbill"
    tb.to_csv(DESTINO)
    print(tb.describe().round(2).to_string())
