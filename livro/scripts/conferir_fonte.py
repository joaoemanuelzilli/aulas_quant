"""Confere os fechamentos do Yahoo contra os oficiais da B3.

Duas fontes do mesmo preço deveriam dar o mesmo retorno em
todos os dias. Este módulo mede onde isso não acontece:

1. comparar: põe as duas fontes lado a lado, um papel por
   vez, e calcula a diferença entre os retornos diários;
2. classificar: separa as diferenças acima de um limite e
   diz de que tipo é cada uma;
3. resumir: conta as divergências por ano e por papel.

O Yahoo entrega a série "na base de hoje": divide os preços
antigos pelos desdobramentos que vieram depois. A B3 traz o
preço negociado no dia. Para comparar, o fechamento da B3 é
levado à escala do Yahoo, trecho por trecho entre dois
eventos societários, e assim o degrau de um desdobramento
não conta como divergência.

Rodado como script, compara os oito papéis de 2010 a 2025.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = Path(os.environ.get("DADOS_LIVRO", RAIZ / "dados"))
BRUTO = DADOS / "bruto"
LIMITE = 0.5     # diferença de retorno que conta, em p.p.
FOLGA = 0.25     # diferença de preço que conta, em %
# Datas em que só o Yahoo muda de escala: ele trata como
# desdobramento a subscrição do Magazine Luiza de 2024.
SO_NO_YAHOO = {"MGLU3": ["2024-02-01"]}


def escala(b3, yahoo, cortes):
    """Razão típica entre B3 e Yahoo em cada trecho.

    Os cortes são as datas ex dos eventos societários. Entre
    dois cortes a razão deveria ser constante; a mediana do
    trecho ignora os dias em que uma das fontes está errada.
    """
    trecho = pd.Series(0, index=b3.index)
    for corte in cortes:
        trecho += b3.index >= pd.Timestamp(corte)
    return (b3 / yahoo).groupby(trecho).transform("median")


def comparar(b3, yahoo, volume, cortes):
    """Compara as duas fontes nos pregões de um papel.

    b3, yahoo: fechamentos sem ajuste por proventos;
    volume: quantidade negociada segundo o Yahoo.
    Devolve uma linha por pregão da B3, com o fechamento
    oficial na escala do Yahoo, os dois retornos, a
    diferença entre eles (dif, em pontos percentuais) e o
    erro de nível do Yahoo (nivel, em %).
    """
    t = pd.DataFrame({"b3": b3.dropna()})
    t["yahoo"] = yahoo.reindex(t.index)
    t["volume"] = volume.reindex(t.index)
    t["oficial"] = t["b3"] / escala(t["b3"], t["yahoo"],
                                    cortes)
    t["ret_b3"] = t["oficial"].pct_change()
    # sem preencher: se falta o dia ou a véspera no Yahoo,
    # não há retorno para comparar
    t["ret_yahoo"] = t["yahoo"].pct_change(fill_method=None)
    t["dif"] = 100 * (t["ret_yahoo"] - t["ret_b3"])
    t["nivel"] = 100 * (t["yahoo"] / t["oficial"] - 1)
    return t


def classificar(t, limite=LIMITE, folga=FOLGA):
    """Divergências acima do limite, com o tipo de cada uma.

    volume zero: o Yahoo registra volume zero (e quase
        sempre repete o preço da véspera);
    repetido: preço igual ao da véspera com volume positivo,
        enquanto a B3 mudou de preço;
    fechamento errado: o preço do dia difere do oficial;
    eco: o preço do dia está certo, mas o da véspera estava
        errado, e o retorno herda o erro com sinal trocado.
    """
    parado = (t["ret_yahoo"] == 0) & (t["ret_b3"] != 0)
    errado = t["nivel"].abs() > folga
    if "papel" in t:      # não passa de um papel a outro
        grupos = errado.groupby(t["papel"])
        vespera = grupos.shift(1, fill_value=False)
    else:
        vespera = errado.shift(1, fill_value=False)
    tipo = np.select(
        [t["volume"] == 0, parado, errado, vespera],
        ["volume zero", "repetido", "fechamento errado",
         "eco"], "outro")
    d = t.assign(tipo=tipo)
    return d[d["dif"].abs() > limite]


def resumir(div, por="papel"):
    """Número de divergências por ano e por papel (ou tipo)."""
    r = pd.crosstab(div.index.year, div[por])
    r.index.name = "ano"
    r["total"] = r.sum(axis=1)
    r.loc["total"] = r.sum()
    return r


def ler_fontes():
    """Fechamentos da B3, do Yahoo, volume e eventos."""
    b3 = pd.read_csv(BRUTO / "b3_cotahist.csv",
                     parse_dates=["data"])
    b3 = b3.pivot(index="data", columns="codigo",
                  values="fechamento")
    ler = lambda nome: pd.read_csv(
        BRUTO / nome, index_col=0, parse_dates=True)
    eventos = pd.read_csv(BRUTO / "eventos_societarios.csv")
    return (b3, ler("precos_sem_ajuste.csv"),
            ler("volume.csv"), eventos)


def conferir(papeis, limite=LIMITE):
    """Compara os papéis; devolve tudo e as divergências."""
    b3, yahoo, volume, eventos = ler_fontes()
    tudo = []
    for p in papeis:
        cortes = list(eventos.loc[eventos["papel"] == p,
                                  "data_ex"])
        cortes += SO_NO_YAHOO.get(p, [])
        t = comparar(b3[p], yahoo[p + ".SA"],
                     volume[p + ".SA"], cortes)
        tudo.append(t.assign(papel=p))
    tudo = pd.concat(tudo)
    return tudo, classificar(tudo, limite)


if __name__ == "__main__":
    papeis = ["ABEV3", "BBAS3", "BOVA11", "ITUB4", "MGLU3",
              "PETR4", "VALE3", "WEGE3"]
    tudo, div = conferir(papeis)
    n = tudo["dif"].notna().sum()
    print(f"retornos comparados: {n}")
    print(f"divergências acima de {LIMITE} p.p.: {len(div)}"
          f" ({len(div) / n:.2%})")
    modulo = div["dif"].abs()
    print(f"mediana: {modulo.median():.2f} p.p. | acima de"
          f" 1 p.p.: {(modulo > 1).sum()} | de 2 p.p.:"
          f" {(modulo > 2).sum()} | de 5 p.p.:"
          f" {(modulo > 5).sum()} | máxima:"
          f" {modulo.max():.1f} p.p.")
    print("\npor tipo:")
    print(div["tipo"].value_counts().to_string())
    print("\npor ano e papel:")
    print(resumir(div).to_string())
    print("\npor ano e tipo:")
    print(resumir(div, "tipo").to_string())
    print("\nas dez maiores:")
    maiores = div.iloc[modulo.values.argsort()[::-1][:10]]
    colunas = ["papel", "b3", "oficial", "yahoo", "ret_b3",
               "ret_yahoo", "dif", "tipo"]
    print(maiores[colunas].round(4).to_string())
