"""Baixa a taxa CDI diária (série 12 do SGS, Banco Central).

Rodado uma vez; o capítulo lê dados/c08_cdi.csv.
"""
import json
import urllib.request

import pandas as pd

URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
       "?formato=json&dataInicial={}&dataFinal={}")
partes = []
# a API devolve no máximo dez anos por consulta
for ini, fim in [("01/01/2010", "31/12/2017"),
                 ("01/01/2018", "31/12/2025")]:
    with urllib.request.urlopen(URL.format(ini, fim)) as resp:
        partes.append(pd.DataFrame(json.load(resp)))
df = pd.concat(partes)
df["Date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
df["cdi"] = df["valor"].astype(float) / 100   # % ao dia -> fração
df = df.set_index("Date")[["cdi"]]
df.to_csv("../../dados/c08_cdi.csv")
print(len(df), "dias;", df.index[0].date(), "a",
      df.index[-1].date())
print(f"CDI acumulado: {(1 + df['cdi']).prod() - 1:.1%}")
