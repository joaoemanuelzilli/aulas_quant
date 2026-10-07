"""Gera os arquivos de preços do livro a partir dos brutos.

Os arquivos em dados/bruto/ são a cópia exata do que o Yahoo
Finance devolveu. Eles têm dois defeitos que contaminam qualquer
estatística:

1. Preço repetido sem negócio. Em feriados da B3 e em vários
   pregões antigos a fonte registra volume zero e repete o último
   fechamento. Esses pontos não são retornos nulos, são dados que
   faltam. Regra: se o volume do papel é zero e o fechamento sem
   ajuste é igual ao último fechamento válido, o ponto vira
   ausente. A comparação usa o preço sem ajuste porque o preço
   ajustado muda nas datas de provento mesmo quando a cotação foi
   apenas repetida.
2. Cisão não ajustada. Em 04/10/2021 os acionistas do Itaú
   Unibanco receberam a participação na XP; a fonte não ajustou o
   histórico e a série mostra uma queda de 18% que não foi
   prejuízo. Os preços anteriores à data são multiplicados por
   (1 - 0,1754), o desconto do evento.

O Capítulo 2 mostra como esses defeitos foram encontrados. Os
demais capítulos leem os arquivos limpos gravados em dados/.
"""
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"
BRUTO = DADOS / "bruto"
DATA_CISAO, FATOR_CISAO = "2021-10-04", 1 - 0.1754


def ler(nome):
    return pd.read_csv(BRUTO / nome, index_col=0,
                       parse_dates=True)


def sem_negocio(preco, volume):
    """Marca pontos com volume zero e preço repetido."""
    marca = pd.Series(False, index=preco.index)
    ultimo = float("nan")           # último fechamento válido
    for dia, p in preco.items():
        if volume.get(dia) == 0 and p == ultimo:
            marca[dia] = True
        elif pd.notna(p):
            ultimo = p
    return marca


if __name__ == "__main__":
    tela = ler("precos_sem_ajuste.csv")
    volume = ler("volume.csv")
    acoes = [c for c in tela.columns if c.endswith(".SA")]
    falso = pd.DataFrame(
        {a: sem_negocio(tela[a], volume[a]) for a in acoes})
    feriado = falso.all(axis=1)     # nenhum papel negociou

    for nome in ["precos_ajustados.csv", "precos_sem_ajuste.csv"]:
        p = ler(nome)
        p[acoes] = p[acoes].mask(falso)
        p.loc[feriado, "^BVSP"] = float("nan")
        antes = p.index < DATA_CISAO
        p.loc[antes, "ITUB4.SA"] *= FATOR_CISAO
        p.to_csv(DADOS / nome)

    ohlc = ler("petr4_ohlcv.csv")
    fora = falso["PETR4.SA"].reindex(ohlc.index, fill_value=False)
    ohlc[~fora].to_csv(DADOS / "petr4_ohlcv.csv")

    print("pontos sem negócio removidos, por papel:")
    print(falso.sum().to_string())
    print("datas que não são pregão:", int(feriado.sum()))
