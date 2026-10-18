"""Testes de sanidade da conferência e da base do livro."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")                 # livro/scripts/
from conferir_fonte import classificar, comparar

# 1. caso feito à mão: um fechamento errado no dia 2 (com eco
#    no dia 3) e um preço repetido com volume zero no dia 5
dias = pd.date_range("2024-01-01", periods=7, freq="B")
b3 = pd.Series([10.0, 10.2, 10.1, 10.3, 10.6, 10.5, 10.4], dias)
yahoo = pd.Series([10.0, 10.2, 9.5, 10.3, 10.3, 10.5, 10.4], dias)
volume = pd.Series([9, 9, 9, 9, 0, 9, 9], dias)
div = classificar(comparar(b3, yahoo, volume, cortes=[]))
esperado = ["fechamento errado", "eco", "volume zero", "eco"]
assert div["tipo"].tolist() == esperado, div
assert div.index.tolist() == list(dias[[2, 3, 4, 5]])

# 2. fontes iguais não geram divergência, mesmo com um
#    desdobramento de 1 para 2 no meio (corte no dia 4)
b3_desd = b3.copy()
b3_desd[dias < dias[4]] *= 2
igual = classificar(comparar(b3_desd, b3, volume + 1,
                             cortes=[dias[4]]))
assert igual.empty, igual

# 3. a base do livro: nos pregões, o retorno do preço sem
#    ajuste é o do fechamento oficial, nos oito papéis, fora
#    das datas de evento (e da cisão de ITUB4)
oficial = pd.read_csv("../../dados/bruto/b3_cotahist.csv",
                      parse_dates=["data"]).pivot(
    index="data", columns="codigo", values="fechamento")
oficial["ABEV3"] = oficial["ABEV3"].fillna(oficial["AMBV3"])
sem = pd.read_csv("../../dados/precos_sem_ajuste.csv",
                  index_col=0, parse_dates=True)
eventos = pd.read_csv("../../dados/bruto/eventos_societarios.csv",
                      parse_dates=["data_ex"])
for papel in ["ABEV3", "BBAS3", "BOVA11", "ITUB4", "MGLU3",
              "PETR4", "VALE3", "WEGE3"]:
    r_of = oficial[papel].dropna().pct_change()
    r_livro = sem[papel + ".SA"].dropna().pct_change()
    ex = list(eventos.loc[eventos["papel"] == papel, "data_ex"])
    if papel == "ITUB4":
        ex.append(pd.Timestamp("2021-10-04"))     # cisão
    comuns = r_of.index.difference(ex)
    assert np.allclose(r_of[comuns], r_livro.reindex(comuns),
                       equal_nan=True, atol=1e-9), papel

print("3 testes passaram")
