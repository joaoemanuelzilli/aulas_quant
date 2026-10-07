"""Motor de backtest vetorizado usado em todo o Capítulo 8."""
import numpy as np
import pandas as pd

DADOS = "../../dados/"
PCT = ("retorno a.a.", "vol a.a.", "drawdown máx.")


def cdi_entre(datas):
    """CDI acumulado entre cada pregão e o anterior."""
    c = pd.read_csv(DADOS + "c08_cdi.csv",
                    index_col=0, parse_dates=True)["cdi"]
    # índice do CDI: a taxa do dia d rende de d até d+1
    ind = (1 + c).cumprod().shift(1, fill_value=1.0)
    return ind.reindex(datas, method="ffill").pct_change()


def carregar(*ativos):
    """Preços, retornos simples e CDI nas mesmas datas."""
    p = pd.read_csv(DADOS + "precos_ajustados.csv",
                    index_col=0, parse_dates=True)
    p = p[list(ativos)].dropna()       # só datas com todos
    cdi = cdi_entre(p.index)
    return p.iloc[1:], p.pct_change().iloc[1:], cdi.iloc[1:]


def carregar_ohlc():
    """Barras diárias de PETR4, retornos e CDI."""
    o = pd.read_csv(DADOS + "petr4_ohlcv.csv",
                    index_col=0, parse_dates=True)
    ret, cdi = o["Close"].pct_change(), cdi_entre(o.index)
    return o.iloc[1:], ret.iloc[1:], cdi.iloc[1:]


def backtest(alvo, ret, cdi, custo=0.001, defasagem=1):
    """Retorno diário líquido de custos de uma regra.

    alvo: fração do capital desejada em cada ativo, decidida
    com a informação disponível no fechamento de cada dia.
    """
    if isinstance(alvo, pd.Series):
        alvo, ret = alvo.to_frame("a"), ret.to_frame("a")
    pos = alvo.shift(defasagem).fillna(0.0)  # w de t-1 rende em t
    bruto = cdi + (pos * ret.sub(cdi, axis=0)).sum(axis=1)
    # peso de ontem depois de derivar com os preços de ontem
    antes = (pos * (1 + ret)).div(1 + bruto, axis=0).shift(1)
    giro = (pos - antes.fillna(0.0)).abs().sum(axis=1)
    return bruto - custo * giro, giro


def metricas(r, cdi):
    """Métricas do Capítulo 7 para uma série de retornos."""
    ex = r - cdi
    curva = (1 + r).cumprod()
    pico = curva.cummax().clip(lower=1.0)  # capital inicial 1
    dd = curva / pico - 1
    fundo = dd < 0                         # abaixo do pico
    ret = curva.iloc[-1] ** (252 / len(r)) - 1
    queda = np.sqrt((ex.clip(upper=0) ** 2).mean())
    return pd.Series({
        "retorno a.a.": ret,
        "vol a.a.": r.std() * np.sqrt(252),
        "Sharpe": ex.mean() / ex.std() * np.sqrt(252),
        "Sortino": ex.mean() / queda * np.sqrt(252),
        "drawdown máx.": dd.min(),
        "submerso (pregões)": fundo.groupby((~fundo).cumsum())
                                   .sum().max(),
        "Calmar": ret / -dd.min(),
    })


def operacoes(alvo, ret, cdi, custo=0.001):
    """Resultado de cada operação de uma regra de um ativo.

    Operação: trecho contínuo com posição de mesmo sinal.
    Resultado: retorno acima do CDI, menos entrada e saída.
    """
    pos = alvo.shift(1).fillna(0.0)
    lado = np.sign(pos)
    bloco = (lado != lado.shift()).cumsum()[lado != 0]
    bruto = 1 + pos * (ret - cdi)
    return bruto.groupby(bloco).prod() - 1 - 2 * custo


def tabela(casos, cdi):
    """Métricas lado a lado, formatadas para impressão."""
    t = pd.DataFrame({k: metricas(r, cdi.loc[r.index])
                      for k, r in casos.items()})
    out = t.astype(object)
    for linha in t.index:
        if linha in PCT:
            out.loc[linha] = [f"{v:.1%}" for v in t.loc[linha]]
        elif linha == "submerso (pregões)":
            out.loc[linha] = [f"{v:.0f}" for v in t.loc[linha]]
        else:
            out.loc[linha] = [f"{v:.2f}" for v in t.loc[linha]]
    return out
