"""Projeto-exemplo do Capítulo 12: momentum em sete ações."""
from pathlib import Path

import numpy as np
import pandas as pd

DADOS = Path(__file__).resolve().parents[2] / "dados"
ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
         "ABEV3.SA", "BBAS3.SA", "MGLU3.SA"]
TREINO = ("2013-01-01", "2019-12-31")
TESTE = ("2020-01-01", "2025-12-31")
CUSTO = 0.001          # 0,10% do valor negociado, por ponta
GRADE_J, GRADE_K = [3, 6, 9, 12], [2, 3, 4]   # 12 variantes


def carregar():
    """Preços ajustados, retornos diários e CDI alinhados."""
    ler = lambda arq: pd.read_csv(DADOS / arq, index_col=0,
                                  parse_dates=True)
    px = ler("precos_ajustados.csv")[ACOES + ["BOVA11.SA"]]
    px = px.dropna(how="all")       # fica só com pregões da B3
    # CDI acumulado de um pregão ao seguinte: a taxa datada
    # do dia d rende de d a d+1, e há dias úteis sem pregão
    fator = (1 + ler("c12_cdi.csv")["cdi"]).cumprod()
    fator = fator.shift(1, fill_value=1.0)
    cdi = fator.reindex(px.index, method="ffill").pct_change()
    return px, px.pct_change(), cdi.fillna(0.0)


def fins_de_mes(px):
    """Linhas do último pregão de cada mês."""
    return px.groupby(px.index.to_period("M")).tail(1)


def alvo_momentum(px, j, k, acoes=ACOES):
    """Pesos-alvo: 1/k nos k papéis de maior alta em j meses."""
    fim = fins_de_mes(px[acoes])
    mom = fim / fim.shift(j) - 1        # só preços até o sinal
    posto = mom.rank(axis=1, ascending=False)
    alvo = (posto <= k).astype(float) / k
    return alvo[mom.notna().all(axis=1)]


def alvo_fixo(px, colunas):
    """Pesos iguais nas colunas dadas, repostos todo mês."""
    fim = fins_de_mes(px)
    return pd.DataFrame(1 / len(colunas), fim.index, colunas)


def simular(alvo, ret, cdi, periodo, custo=CUSTO, atraso=1):
    """Carteira dia a dia: retorno líquido, posições e giro."""
    ini, fim = periodo
    # atraso=1: o sinal de um pregão vira ordem no fechamento
    # do seguinte; atraso=0: no fechamento do próprio sinal
    ordens = alvo.reindex(ret.index).shift(atraso)
    antes = ordens[ordens.index < ini].dropna()  # já emitidas
    ordens = ordens.loc[ini:fim].copy()
    # a rodada começa em dinheiro e compra a carteira vigente
    # no fechamento do primeiro pregão, com qualquer atraso
    if len(antes) and ordens.iloc[0].isna().any():
        ordens.iloc[0] = antes.iloc[-1]
    r = ret.loc[ini:fim, alvo.columns].fillna(0.0)
    caixa = cdi.loc[ini:fim].to_numpy()
    w = np.zeros(r.shape[1])
    liquido, posicoes, giro = [], [], []
    for r_dia, c_dia, ordem in zip(r.to_numpy(), caixa,
                                   ordens.to_numpy()):
        # o que não está em ações rende o CDI do dia
        bruto = w @ r_dia + (1 - w.sum()) * c_dia
        w = w * (1 + r_dia) / (1 + bruto)     # pesos derivam
        negociado = 0.0
        if not np.isnan(ordem).any():         # dia de ordem
            negociado = np.abs(ordem - w).sum()
            w = ordem
        liquido.append((1 + bruto) * (1 - custo * negociado) - 1)
        posicoes.append(w)
        giro.append(negociado)
    return (pd.Series(liquido, r.index),
            pd.DataFrame(posicoes, r.index, alvo.columns),
            pd.Series(giro, r.index))


def operacoes(pos, px, custo=CUSTO):
    """Operação: da entrada à saída de um papel."""
    linhas = []
    for a in pos.columns:
        dentro = (pos[a] > 0).astype(int)
        muda = dentro.diff().fillna(dentro)
        entradas = list(muda.index[muda == 1])
        saidas = list(muda.index[muda == -1]) + [pos.index[-1]]
        for e, s in zip(entradas, saidas):
            res = px.at[s, a] / px.at[e, a] - 1 - 2 * custo
            linhas.append((a, e, s, res))
    return pd.DataFrame(linhas, columns=["ativo", "entrada",
                                         "saida", "retorno"])


def drawdown(ret):
    """Queda da cota em relação ao maior valor já atingido."""
    cota = (1 + ret).cumprod()
    return cota / cota.cummax().clip(lower=1.0) - 1


def metricas(ret, cdi):
    """Métricas obrigatórias, a partir do retorno diário."""
    cota = (1 + ret).cumprod()
    anos = len(ret) / 252               # convenção de 252 dias
    excesso = ret - cdi.reindex(ret.index)
    dd = drawdown(ret)
    vale = dd.idxmin()
    pico = dd.loc[:vale][dd.loc[:vale] == 0].index
    queda = dd.loc[pico[-1]:vale].size - 1 if len(pico) else (
        dd.loc[:vale].size)             # pico no capital inicial
    apos = dd.loc[vale:]
    volta = apos[apos == 0].index       # cota de volta ao pico
    rec = apos.loc[:volta[0]].size - 1 if len(volta) else np.nan
    return pd.Series({
        "retorno acumulado": cota.iloc[-1] - 1,
        "retorno a.a.": cota.iloc[-1] ** (1 / anos) - 1,
        "vol a.a.": ret.std() * np.sqrt(252),
        "Sharpe": excesso.mean() / excesso.std() * np.sqrt(252),
        "drawdown máx.": dd.min(),
        "queda (pregões)": queda,
        "recuperação (pregões)": rec,
    })


def ativo_mensal(ret, ref):
    """Retorno ativo mensal, razão de informação e t."""
    mes = lambda s: s.index.to_period("M")
    a = ((1 + ret).groupby(mes(ret)).prod()
         - (1 + ref).groupby(mes(ref)).prod())
    ri = a.mean() / a.std() * np.sqrt(12)
    return a, ri, a.mean() / a.std() * np.sqrt(len(a))
