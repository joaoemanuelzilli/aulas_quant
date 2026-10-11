"""Gera os arquivos de preços do livro a partir dos brutos.

A pasta dados/bruto/ guarda duas fontes, nunca editadas:

- b3_cotahist.csv: cotações oficiais da B3 (abertura, máxima,
  mínima, fechamento, negócios, quantidade e volume), tiradas
  dos arquivos anuais da bolsa por extrair_cotahist.py;
- os arquivos baixados do Yahoo Finance por baixar_dados.py.

O preço de cada ação vem da B3. O fechamento oficial é o
preço negociado no dia e precisa de três correções para virar
uma série comparável no tempo:

1. Eventos societários. Num desdobramento de 1 para 2 a
   cotação cai pela metade sem que ninguém perca nada. A
   tabela eventos_societarios.csv traz, para cada evento, a
   data ex e o fator que multiplica todos os preços
   anteriores a ela, escrito como fração: ações antes sobre
   ações depois (1/2 no exemplo).
2. Cisão. Em 04/10/2021 os acionistas do Itaú Unibanco
   receberam a participação na XP. Os preços anteriores de
   ITUB4 são multiplicados por (1 - 0,1754).
3. Proventos. O fator acumulado de dividendos e juros sobre
   capital próprio é a razão entre o fechamento ajustado e o
   fechamento sem ajuste do Yahoo, que só muda nas datas ex.
   Onde o Yahoo erra o valor ou a data de um provento, vale
   o valor oficial da tabela correcoes_proventos.csv.

precos_sem_ajuste.csv recebe as correções 1 e 2;
precos_ajustados.csv, as três. Do Yahoo continuam vindo o
Ibovespa, o S&P 500 e o dólar. Em dia sem pregão da B3 as
ações e o Ibovespa ficam ausentes: nenhum preço é repetido.
"""
import os
from fractions import Fraction
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = Path(os.environ.get("DADOS_LIVRO", RAIZ / "dados"))
BRUTO = DADOS / "bruto"
DATA_CISAO, FATOR_CISAO = "2021-10-04", 1 - 0.1754
# a Ambev negociou como AMBV3 até 08/11/2013
CODIGO_ANTIGO = {"AMBV3": "ABEV3"}
SALTO = 1e-4          # menor variação que conta como provento
CASAS = 10            # casas decimais gravadas nos preços


def ler(nome):
    """Lê um arquivo bruto do Yahoo, com datas no índice.

    A opção round_trip faz o número lido ser exatamente o
    que está escrito, e as colunas copiadas saem idênticas.
    """
    return pd.read_csv(BRUTO / nome, index_col=0,
                       parse_dates=True,
                       float_precision="round_trip")


def ler_b3():
    """Lê o extrato da B3 e unifica o código da Ambev."""
    b3 = pd.read_csv(BRUTO / "b3_cotahist.csv",
                     parse_dates=["data"])
    b3["codigo"] = b3["codigo"].replace(CODIGO_ANTIGO)
    return b3[b3["quantidade"] > 0]


def tabela(b3, campo):
    """Uma linha por pregão, uma coluna por papel."""
    t = b3.pivot(index="data", columns="codigo",
                 values=campo)
    t.index.name, t.columns.name = "Date", None
    return t.add_suffix(".SA")


def fator_eventos(pregoes, papeis):
    """Fator de eventos societários de cada papel e pregão.

    Em cada data, é o produto dos fatores de todos os eventos
    do papel com data ex posterior a ela.
    """
    ev = pd.read_csv(BRUTO / "eventos_societarios.csv",
                     parse_dates=["data_ex"])
    fator = pd.DataFrame(1.0, index=pregoes, columns=papeis)
    for e in ev.itertuples():
        antes = pregoes < e.data_ex
        fator.loc[antes, e.papel + ".SA"] *= float(
            Fraction(e.fator))
    return fator


def fator_proventos(ajustado, sem_ajuste, pregoes):
    """Fator acumulado de proventos de um papel, limpo.

    A razão entre os dois fechamentos do Yahoo deveria mudar
    só nas datas ex. O arredondamento da fonte faz a razão
    tremer na sétima casa; cada trecho entre dois saltos é
    trocado pela sua mediana. Em pregão que o Yahoo não tem,
    vale o fator do pregão anterior.
    """
    razao = (ajustado / sem_ajuste).dropna()
    salto = (razao / razao.shift(1) - 1).abs() > SALTO
    limpo = razao.groupby(salto.cumsum()).transform("median")
    assert limpo.is_monotonic_increasing
    return limpo.reindex(pregoes).ffill()


def corrigir_proventos(fator, fech):
    """Aplica ao fator as correções de proventos conferidas.

    Em cada linha da tabela, o degrau do fator na data ex
    passa a ser 1 - valor oficial / fechamento da véspera,
    com os dois números na base da época (o fechamento é o
    oficial da B3, antes de qualquer ajuste). Valor oficial
    zero desfaz um degrau que o Yahoo pôs na data errada.
    """
    cor = pd.read_csv(BRUTO / "correcoes_proventos.csv",
                      parse_dates=["data_ex"])
    for c in cor.itertuples():
        a, dia = c.papel + ".SA", c.data_ex
        vespera = fech[a].dropna().shift(1)[dia]
        velho = fator[a].shift(1)[dia] / fator.at[dia, a]
        novo = 1 - c.valor_oficial / vespera
        fator.loc[fator.index < dia, a] *= novo / velho
    assert fator.diff().min().min() > -1e-12   # não decresce
    return fator


def precos_b3(b3):
    """Fechamentos oficiais sem ajuste e ajustados."""
    fech = tabela(b3, "fechamento")
    eventos = fator_eventos(fech.index, fech.columns)
    sem = fech * eventos
    sem.loc[sem.index < DATA_CISAO, "ITUB4.SA"] *= FATOR_CISAO
    y_aj = ler("precos_ajustados.csv")
    y_sem = ler("precos_sem_ajuste.csv")
    proventos = pd.DataFrame({
        a: fator_proventos(y_aj[a], y_sem[a], fech.index)
        for a in fech.columns})
    proventos = corrigir_proventos(proventos, fech)
    return sem, sem * proventos, eventos, proventos


def juntar(nome, acoes, indice):
    """Põe as ações da B3 nas colunas do arquivo do Yahoo."""
    p = ler(nome).reindex(indice)
    p[acoes.columns] = acoes.reindex(indice).round(CASAS)
    p.loc[~indice.isin(acoes.index), "^BVSP"] = float("nan")
    return p


def barras(b3, papel, eventos, proventos):
    """Barras diárias oficiais de um papel, ajustadas."""
    b = b3[b3["codigo"] == papel].set_index("data")
    k = eventos[papel + ".SA"].reindex(b.index)
    total = k * proventos[papel + ".SA"].reindex(b.index)
    colunas = {"Close": "fechamento", "High": "maxima",
               "Low": "minima", "Open": "abertura"}
    o = pd.DataFrame({c: (b[v] * total).round(CASAS)
                      for c, v in colunas.items()})
    o["Volume"] = (b["quantidade"] / k).round().astype("int64")
    o.index.name = "Date"
    return o


if __name__ == "__main__":
    b3 = ler_b3()
    sem, ajustado, eventos, proventos = precos_b3(b3)
    indice = ler("precos_sem_ajuste.csv").index.union(sem.index)
    indice.name = "Date"

    for nome, acoes in [("precos_sem_ajuste.csv", sem),
                        ("precos_ajustados.csv", ajustado)]:
        juntar(nome, acoes, indice).to_csv(DADOS / nome)

    barras(b3, "PETR4", eventos, proventos).to_csv(
        DADOS / "petr4_ohlcv.csv")
    tabela(b3, "volume").reindex(indice).to_csv(
        DADOS / "volume_financeiro.csv")

    print("pregões da B3:", len(sem), "| linhas:", len(indice))
    print("valores ausentes em pregão da B3, por papel:")
    print(sem.isna().sum().to_string())
