"""Baixa a taxa CDI diária (série 12 do SGS, Banco Central).

Rode uma única vez; o capítulo lê dados/c07_cdi.csv. A coluna
`cdi` é a taxa do dia em fração (0.0004 = 0,04% ao dia).
"""
from pathlib import Path

import pandas as pd
import requests

URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
       "?formato=json&dataInicial={}&dataFinal={}")
DESTINO = Path(__file__).resolve().parents[2] / "dados"

# a API limita séries diárias a dez anos por consulta
janelas = [("01/01/2010", "31/12/2017"),
           ("01/01/2018", "31/12/2025")]
partes = []
for ini, fim in janelas:
    dados = requests.get(URL.format(ini, fim), timeout=60).json()
    partes.append(pd.DataFrame(dados))
df = pd.concat(partes)
df["Date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
df["cdi"] = df["valor"].astype(float) / 100
df = df.set_index("Date")[["cdi"]].sort_index()
df.to_csv(DESTINO / "c07_cdi.csv")
print(len(df), df.index[0].date(), df.index[-1].date())
anual = (1 + df["cdi"]).groupby(df.index.year).prod() - 1
print(anual.round(4))
