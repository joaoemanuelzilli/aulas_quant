"""Backtest vetorizado de cruzamento de médias móveis."""
import pandas as pd

DADOS = "../../dados/"


def carregar(ativo="PETR4.SA", mercado="^BVSP"):
    """Retornos diários do ativo, do mercado e do CDI."""
    px = pd.read_csv(DADOS + "precos_ajustados.csv",
                     index_col=0, parse_dates=True)
    cdi = pd.read_csv(DADOS + "c07_cdi.csv",
                      index_col=0, parse_dates=True)["cdi"]
    preco = px[ativo].dropna()
    dias = preco.index
    # índice do CDI: a taxa do dia d rende de d até d+1
    caixa = (1 + cdi).cumprod().shift(1, fill_value=1.0)
    caixa = caixa.reindex(dias, method="ffill")
    indice = px[mercado].reindex(dias).ffill()
    ret = pd.DataFrame({"ativo": preco.pct_change(),
                        "mercado": indice.pct_change(),
                        "cdi": caixa.pct_change()})
    return preco.iloc[1:], ret.iloc[1:]


def cruzamento(preco, cdi, rapida=20, lenta=100,
               custo=0.001, defasagem=1):
    """Comprado quando a média rápida supera a lenta."""
    media_r = preco.rolling(rapida).mean()
    media_l = preco.rolling(lenta).mean()
    # o sinal de t só é conhecido no fechamento de t
    sinal = (media_r > media_l).astype(float)
    # por isso a posição do dia t usa o sinal de t - defasagem
    posicao = sinal.shift(defasagem).fillna(0.0)
    giro = posicao.diff().abs().fillna(0.0)  # fração negociada
    bruto = posicao * preco.pct_change().fillna(0.0)
    caixa = (1 - posicao) * cdi              # fora da ação: CDI
    return pd.DataFrame({"posicao": posicao, "giro": giro,
                         "retorno": bruto + caixa - custo * giro})
