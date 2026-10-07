"""Funções de carteira usadas nos scripts do Capítulo 6."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DADOS = "../../dados/"
ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
         "ABEV3.SA", "BBAS3.SA", "MGLU3.SA"]


def retornos(inicio, fim, ativos=ACOES):
    """Retornos simples diários das ações do livro. Dias em
    que algum ativo não tem preço são descartados; o retorno
    seguinte cobre o intervalo inteiro."""
    p = pd.read_csv(DADOS + "precos_ajustados.csv",
                    index_col=0, parse_dates=True)
    p = p[ativos].dropna()
    return p.pct_change().dropna().loc[inicio:fim]


def cdi_anual(inicio, fim):
    """CDI do período: média diária vezes 252, na mesma base
    da média dos retornos devolvida por `insumos`."""
    cdi = pd.read_csv(DADOS + "c06_cdi.csv", index_col=0,
                      parse_dates=True)["cdi"].loc[inicio:fim]
    return cdi.mean() * 252


def insumos(r):
    """Média e covariância anualizadas (252 dias úteis)."""
    return r.mean().values * 252, r.cov().values * 252


def vol(w, cov):
    return np.sqrt(w @ cov @ w)


SOMA_1 = {"type": "eq", "fun": lambda w: w.sum() - 1}


def otimizar(objetivo, n, restricoes=(), teto=1.0):
    """Minimiza `objetivo` com soma 1 e pesos entre 0 e teto."""
    res = minimize(objetivo, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, teto)] * n,
                   constraints=[SOMA_1, *restricoes],
                   options={"ftol": 1e-12, "maxiter": 500})
    assert res.success, res.message
    return res.x


def minima_variancia(cov, teto=1.0):
    return otimizar(lambda w: w @ cov @ w, len(cov), teto=teto)


def maximo_sharpe(mu, cov, rf, teto=1.0):
    def menos_sharpe(w):
        return -(w @ mu - rf) / vol(w, cov)
    return otimizar(menos_sharpe, len(mu), teto=teto)


def fronteira(mu, cov, alvos):
    """Menor volatilidade para cada retorno-alvo (sem venda
    a descoberto)."""
    vols = []
    for alvo in alvos:
        no_alvo = {"type": "eq",
                   "fun": lambda w, a=alvo: w @ mu - a}
        w = otimizar(lambda w: w @ cov @ w, len(mu), [no_alvo])
        vols.append(vol(w, cov))
    return np.array(vols)
