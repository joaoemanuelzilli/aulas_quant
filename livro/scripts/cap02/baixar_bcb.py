"""Baixa CDI e dólar PTAX do SGS (Banco Central) e congela."""
from pathlib import Path

import pandas as pd
import requests

DADOS = Path("../../dados")
URL = ("https://api.bcb.gov.br/dados/serie/"
       "bcdata.sgs.{codigo}/dados")
SERIES = {"cdi": 12, "ptax_venda": 1}     # nome: código no SGS
JANELAS = [("01/01/2010", "31/12/2017"),  # o SGS limita a 10 anos
           ("01/01/2018", "31/12/2025")]  # por consulta diária


def baixar(codigo):
    partes = []
    for inicio, fim in JANELAS:
        resposta = requests.get(
            URL.format(codigo=codigo), timeout=30,
            params={"formato": "json", "dataInicial": inicio,
                    "dataFinal": fim})
        resposta.raise_for_status()       # erro visível
        partes.append(pd.DataFrame(resposta.json()))
    bruto = pd.concat(partes)
    datas = pd.to_datetime(bruto["data"], format="%d/%m/%Y")
    return pd.Series(bruto["valor"].astype(float).values,
                     index=datas.rename("Date"))


arquivo = DADOS / "c02_bcb.csv"
if not arquivo.exists():                  # baixa uma única vez
    series = {nome: baixar(c) for nome, c in SERIES.items()}
    pd.DataFrame(series).to_csv(arquivo)

bcb = pd.read_csv(arquivo, index_col=0, parse_dates=True)
print(bcb.loc["2024-01-02":"2024-01-04"])
cdi_2024 = (1 + bcb.loc["2024", "cdi"] / 100).prod() - 1
print(f"dias: {len(bcb)}   CDI acumulado em 2024: {cdi_2024:.2%}")
