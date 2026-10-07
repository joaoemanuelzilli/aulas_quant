"""Baixa o CDI diário (série 12 do SGS/Banco Central) uma vez
e grava em dados/c09_cdi.csv, em fração ao dia."""
import pandas as pd

URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
       "?formato=json&dataInicial={}&dataFinal={}")
# a API limita séries diárias a janelas de dez anos
janelas = [("01/01/2010", "31/12/2017"),
           ("01/01/2018", "31/12/2025")]
partes = [pd.read_json(URL.format(a, b)) for a, b in janelas]
cdi = pd.concat(partes)
cdi.index = pd.to_datetime(cdi.pop("data"), format="%d/%m/%Y")
cdi.index.name = "Date"
cdi = (cdi["valor"] / 100).rename("cdi")   # % ao dia -> fração
cdi.to_csv("../../dados/c09_cdi.csv")
print(len(cdi), cdi.index[0].date(), cdi.index[-1].date())
print(f"CDI acumulado: {(1 + cdi).prod() - 1:.1%}")
