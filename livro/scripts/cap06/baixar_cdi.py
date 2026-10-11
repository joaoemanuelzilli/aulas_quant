"""Baixa a taxa CDI diária (série 12 do SGS/Banco Central).

Rode uma única vez; grava dados/c06_cdi.csv com a taxa em
fração ao dia (0.000437 = 0,0437% a.d.). O SGS limita as
consultas de séries diárias a dez anos, daí os dois pedidos.
"""
from pathlib import Path

import pandas as pd
import requests

URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
       "?formato=json&dataInicial={}&dataFinal={}")
DESTINO = Path(__file__).resolve().parents[2] / "dados"

partes = []
for ini, fim in [("01/01/2010", "31/12/2017"),
                 ("01/01/2018", "31/12/2025")]:
    resposta = requests.get(URL.format(ini, fim), timeout=60)
    resposta.raise_for_status()
    partes.append(pd.DataFrame(resposta.json()))

cdi = pd.concat(partes)
cdi["data"] = pd.to_datetime(cdi["data"], format="%d/%m/%Y")
cdi = cdi.set_index("data")["valor"].astype(float) / 100
cdi = cdi.round(8)                # tira o ruído da divisão
cdi.name = "cdi"
cdi.to_csv(DESTINO / "c06_cdi.csv")

anual = (1 + cdi).groupby(cdi.index.year).prod() - 1
print(anual.map(lambda v: f"{v:.2%}").to_string())
