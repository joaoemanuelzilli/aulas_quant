"""Painel de features e rótulo usado em todo o capítulo."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
         "ABEV3.SA", "BBAS3.SA", "MGLU3.SA"]
H = 5                      # horizonte do rótulo, em pregões
CORTE = "2022-01-01"       # daqui em diante: período intocado
FEATS = ["ret_1", "ret_5", "ret_21", "ret_63", "vol_21",
         "vol_razao", "pm_50", "rsi_14", "dd_252", "ibov_21"]


def rsi(p, n=14):
    """RSI com média exponencial de Wilder, sem a semente de média
    simples do Capítulo 9; depois dos 252 pregões de aquecimento
    do painel a diferença entre as duas versões é desprezível."""
    d = p.diff()
    media = lambda s: s.ewm(alpha=1 / n, min_periods=n).mean()
    ganho = media(d.clip(lower=0))
    perda = media((-d).clip(lower=0))
    return 100 - 100 / (1 + ganho / perda)


def features(p, merc_21):
    """Tudo calculado só com preços até o fechamento de t."""
    r = p.pct_change()
    f = pd.DataFrame(index=p.index)
    f["ret_1"] = r
    f["ret_5"] = p.pct_change(5)
    f["ret_21"] = p.pct_change(21)
    f["ret_63"] = p.pct_change(63)
    f["vol_21"] = r.rolling(21).std()
    f["vol_razao"] = f["vol_21"] / r.rolling(126).std()
    f["pm_50"] = p / p.rolling(50).mean() - 1
    f["rsi_14"] = rsi(p)
    f["dd_252"] = p / p.rolling(252).max() - 1
    f["ibov_21"] = merc_21
    return f


def painel():
    precos = pd.read_csv("../../dados/precos_ajustados.csv",
                         index_col=0, parse_dates=True)
    merc_21 = precos["BOVA11.SA"].dropna().pct_change(21)
    blocos = []
    for a in ACOES:
        p = precos[a].dropna()     # MGLU3 só existe desde 2011
        f = features(p, merc_21.reindex(p.index))
        f["fwd"] = p.shift(-H) / p - 1     # retorno de t a t+H
        f["ret_seg"] = p.shift(-1) / p - 1  # retorno de t a t+1
        f["acao"] = a
        blocos.append(f)
    d = pd.concat(blocos).dropna().sort_index(kind="stable")
    d["y"] = (d["fwd"] > 0).astype(int)   # 1 se subiu em H dias
    return d


def dev_teste(d):
    """Desenvolvimento e teste final, com H dias de folga."""
    dias = d.index.unique()
    fim_dev = dias[dias < CORTE][-H - 1]
    return d[d.index <= fim_dev], d[d.index >= CORTE]


def cortes(datas, n=5, folga=H):
    """Divisões temporais por data (várias linhas por dia)."""
    dias = np.sort(datas.unique())
    tss = TimeSeriesSplit(n_splits=n, gap=folga)
    return [(np.flatnonzero(datas.isin(dias[tr])),
             np.flatnonzero(datas.isin(dias[te])))
            for tr, te in tss.split(dias)]
