"""Gera os arquivos de preços usados no livro a partir dos arquivos brutos.

Os arquivos em dados/bruto/ são a cópia exata do que o Yahoo Finance
devolveu. Eles têm dois defeitos que contaminam qualquer estatística:

1. Preço repetido sem negócio. Em feriados da B3 e em vários pregões
   antigos a fonte registra volume zero e repete o fechamento anterior.
   Esses pontos não são retornos nulos, são dados que faltam. Regra: se
   o volume do papel é zero e o preço não mudou, o preço vira ausente.
2. Cisão não ajustada. Em 04/10/2021 o Itaú Unibanco entregou aos
   acionistas as ações da XPart; a fonte não ajustou o histórico e a
   série mostra uma queda de 18% que não foi prejuízo. Os preços
   anteriores à data são multiplicados por (1 - 0,1754), o desconto
   do evento.

O Capítulo 2 mostra como esses defeitos foram encontrados. Os demais
capítulos leem os arquivos limpos gravados aqui, em dados/.
"""
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"
BRUTO = DADOS / "bruto"
DATA_CISAO, FATOR_CISAO = "2021-10-04", 1 - 0.1754


def ler(nome):
    return pd.read_csv(BRUTO / nome, index_col=0, parse_dates=True)


def sem_negocio(precos, volume):
    """Marca (data, papel) com volume zero e preço igual ao anterior."""
    acoes = [c for c in precos.columns if c.endswith(".SA")]
    parado = precos[acoes].pct_change(fill_method=None).abs() < 1e-12
    return parado & (volume[acoes].reindex(precos.index) == 0)


if __name__ == "__main__":
    volume = ler("volume.csv")
    for nome in ["precos_ajustados.csv", "precos_sem_ajuste.csv"]:
        p = ler(nome)
        falso = sem_negocio(p, volume)
        p[falso.columns] = p[falso.columns].mask(falso)
        # datas em que nenhum papel negociou não são pregão da B3
        feriado = falso.all(axis=1)
        p.loc[feriado, "^BVSP"] = float("nan")
        if nome == "precos_ajustados.csv":
            antes = p.index < DATA_CISAO
            p.loc[antes, "ITUB4.SA"] *= FATOR_CISAO
            print("pontos sem negócio removidos, por papel:")
            print(falso.sum().to_string())
            print("datas que não são pregão:", int(feriado.sum()))
        p.to_csv(DADOS / nome)

    ohlc = ler("petr4_ohlcv.csv")
    parado = (ohlc["Volume"] == 0) & (ohlc["Close"].diff().abs() < 1e-12)
    ohlc[~parado].to_csv(DADOS / "petr4_ohlcv.csv")
    print("barras de PETR4 sem negócio removidas:", int(parado.sum()))
