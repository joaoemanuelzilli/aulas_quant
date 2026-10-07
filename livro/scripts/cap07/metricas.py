"""Métricas de desempenho a partir de retornos diários."""
import numpy as np

DIAS = 252


def retorno_anual(r):
    return (1 + r).prod() ** (DIAS / len(r)) - 1


def vol_anual(r):
    return r.std() * np.sqrt(DIAS)


def drawdown_maximo(r):
    curva = (1 + r).cumprod()
    pico = curva.cummax().clip(lower=1.0)    # capital inicial 1
    return (curva / pico - 1).min()


def calmar(r):
    return retorno_anual(r) / abs(drawdown_maximo(r))


def sharpe(r, rf):
    excesso = r - rf
    return excesso.mean() / excesso.std() * np.sqrt(DIAS)


def sortino(r, alvo):
    desvio = r - alvo
    abaixo = np.minimum(desvio, 0.0)         # zero nos dias bons
    downside = np.sqrt((abaixo ** 2).mean())  # divide por T
    return desvio.mean() / downside * np.sqrt(DIAS)


def beta_alfa(r, mercado, rf):
    y, x = r - rf, mercado - rf
    beta = np.cov(y, x)[0, 1] / x.var()      # inclinação de MQO
    alfa = (y.mean() - beta * x.mean()) * DIAS
    return beta, alfa


def treynor(r, mercado, rf):
    beta, _ = beta_alfa(r, mercado, rf)
    return (r - rf).mean() * DIAS / beta


def information_ratio(r, referencia):
    ativo = r - referencia
    return ativo.mean() / ativo.std() * np.sqrt(DIAS)
