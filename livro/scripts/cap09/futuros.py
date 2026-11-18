"""Funções do Capítulo 9: leitura, vencimentos e séries mensais.

A EIA publica o "contrato 1" (o de vencimento mais próximo) até
o 4. Para saber qual mês de entrega cada coluna representa em cada
data, basta conhecer a regra de vencimento de cada produto.
"""
import numpy as np
import pandas as pd

DADOS = "../../dados/c09_eia.csv"
FIM_FUTUROS = "2024-03-31"      # último fim de mês com futuros na EIA


def ler(colunas=None, inicio=None, fim=None):
    """Lê dados/c09_eia.csv (preços da EIA, ver baixar_eia.py)."""
    d = pd.read_csv(DADOS, index_col=0, parse_dates=True)
    if colunas is not None:
        d = d[list(colunas)]
    return d.loc[inicio:fim].dropna(how="all")


def vencimento(mes, uteis, produto):
    """Último pregão do contrato com entrega no mês `mes` (Period).

    `uteis` é o índice de pregões. WTI (NYMEX): três dias úteis
    antes do dia 25 do mês anterior à entrega (se o 25 não for
    útil, conta-se a partir do dia útil anterior a ele). Gás:
    três dias úteis antes do 1º dia do mês de entrega.
    """
    anterior = (mes - 1).to_timestamp()
    if produto == "cl":
        d25 = anterior + pd.Timedelta(days=24)
        antes = uteis[uteis < d25]
        if d25 not in uteis:            # 25 não útil: recua mais um
            antes = antes[:-1]
        return antes[-3]
    antes = uteis[uteis < mes.to_timestamp()]
    return antes[-3]


def mes_do_contrato(data, k, uteis, produto):
    """Mês de entrega do contrato k (1 a 4) na data `data`."""
    mes = data.to_period("M") + 1
    while vencimento(mes, uteis, produto) < data:
        mes += 1
    return mes + (k - 1)


def fim_de_mes(colunas, produto, inicio="2007-12", fim=FIM_FUTUROS):
    """Último pregão de cada mês em que todas as colunas têm preço,
    escolhido depois do vencimento do mês: assim o contrato 2 de um
    fim de mês é o contrato 1 do fim do mês seguinte."""
    d = ler(colunas, inicio, fim).dropna()
    uteis = ler([f"{produto}1"]).index
    uteis = uteis.append(pd.bdate_range(uteis[-1], periods=60)[1:])
    meses = d.index.to_period("M")
    venc = {p: vencimento(p + 1, uteis, produto) for p in meses.unique()}
    depois = d.index > pd.DatetimeIndex([venc[p] for p in meses])
    m = d[depois].groupby(meses[depois]).tail(1)
    p = m.index.to_period("M")
    assert (p[1:] - p[:-1] == p.freq).all(), "falta um mês"
    return m
