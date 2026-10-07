"""Baixa uma única vez o CDI diário (série 12 do SGS, BCB).

Grava dados/c12_cdi.csv com a taxa de cada dia útil em fração
(0.0004 = 0,04% ao dia). A API limita cada consulta de série
diária a dez anos, por isso o período é pedido em dois blocos.
"""
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parents[2] / "dados"
URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
       "?formato=json&dataInicial={}&dataFinal={}")
blocos = [("01/01/2010", "31/12/2017"),
          ("01/01/2018", "31/12/2025")]

partes = [pd.read_json(URL.format(a, b)) for a, b in blocos]
cdi = pd.concat(partes)
cdi["data"] = pd.to_datetime(cdi["data"], format="%d/%m/%Y")
cdi = cdi.set_index("data")["valor"].astype(float) / 100
cdi.rename("cdi").to_csv(DADOS / "c12_cdi.csv")
print(len(cdi), cdi.index[0].date(), cdi.index[-1].date())
ano = (1 + cdi).groupby(cdi.index.year).prod() - 1
print(ano.map(lambda v: f"{v:.2%}").to_string())
