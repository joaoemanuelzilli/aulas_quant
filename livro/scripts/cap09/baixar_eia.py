"""Baixa preços diários de petróleo e gás natural da EIA.

Rode uma única vez; grava dados/c09_eia.csv, uma coluna por série,
com corte em 30/09/2026. Se o arquivo já existe, não baixa de novo.

Séries (todas da U.S. Energy Information Administration):
  wti      WTI à vista em Cushing, US$/barril (RWTC)
  brent    Brent à vista FOB Europa, US$/barril (RBRTE)
  hh       gás natural à vista em Henry Hub, US$/MMBtu (RNGWHHD)
  cl1..4   contratos 1 a 4 do futuro de WTI na NYMEX (RCLC1..4)
  ng1..4   contratos 1 a 4 do futuro de gás natural (RNGC1..4)
A EIA não publica os futuros depois de 05/04/2024.
Leitura do .xls pede o pacote xlrd.
"""
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

DESTINO = Path(__file__).resolve().parents[2] / "dados" / "c09_eia.csv"
CORTE = "2026-09-30"
BASE = "https://www.eia.gov/dnav/{}/hist_xls/{}d.xls"
SERIES = {"wti": ("pet", "RWTC"), "brent": ("pet", "RBRTE"),
          "hh": ("ng", "RNGWHHD")}
for k in range(1, 5):
    SERIES[f"cl{k}"] = ("pet", f"RCLC{k}")
    SERIES[f"ng{k}"] = ("ng", f"RNGC{k}")


def baixar(pasta, codigo):
    resposta = requests.get(BASE.format(pasta, codigo), timeout=120)
    resposta.raise_for_status()
    tab = pd.read_excel(BytesIO(resposta.content), sheet_name="Data 1",
                        skiprows=2, index_col=0)
    serie = tab.iloc[:, 0].astype(float)
    serie.index = pd.to_datetime(serie.index)
    return serie


if DESTINO.exists():
    print("já existe:", DESTINO.name, "(nada baixado)")
else:
    dados = pd.DataFrame({nome: baixar(*fonte)
                          for nome, fonte in SERIES.items()})
    dados = dados.sort_index().loc[:CORTE]
    dados.index.name = "data"
    dados.to_csv(DESTINO)
    print(dados.notna().sum().to_string())
