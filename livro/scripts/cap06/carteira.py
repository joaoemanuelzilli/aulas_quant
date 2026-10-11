"""Funções de carteira usadas nos scripts do Capítulo 6."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DADOS = "../../dados/"
ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
         "ABEV3.SA", "BBAS3.SA", "MGLU3.SA"]
NOMES = [a.removesuffix(".SA") for a in ACOES]


def retornos(inicio, fim, ativos=ACOES):
    """Retornos simples diários das ações do livro."""
    p = pd.read_csv(DADOS + "precos_ajustados.csv",
                    index_col=0, parse_dates=True)
    p = p[ativos].dropna()     # tira as datas sem pregão na B3
    return p.pct_change().dropna().loc[inicio:fim]


def carregar_cdi(indice):
    """CDI acumulado entre pregões consecutivos de `indice`."""
    cdi = pd.read_csv(DADOS + "c06_cdi.csv", index_col=0,
                      parse_dates=True)["cdi"]
    assert cdi.max() < 0.01, "CDI deve estar em fração ao dia"
    fator = (1 + cdi).cumprod()
    fator = fator.reindex(fator.index.union(indice)).ffill()
    por_pregao = fator.loc[indice].pct_change()
    return por_pregao.fillna(cdi.mean())   # só o 1º dia


def cdi_anual(inicio, fim):
    """CDI por pregão das sete ações, média vezes 252: a
    mesma base das médias devolvidas por `insumos`."""
    pregoes = retornos(None, fim).index
    return carregar_cdi(pregoes).loc[inicio:fim].mean() * 252


def insumos(r):
    """Média e covariância anualizadas (252 dias úteis)."""
    return r.mean().values * 252, r.cov().values * 252


def vol(w, cov):
    return np.sqrt(w @ cov @ w)


SOMA_1 = {"type": "eq", "fun": lambda w: w.sum() - 1}


def otimizar(objetivo, n, restricoes=(), teto=1.0, w0=None):
    """Minimiza `objetivo` com soma 1 e pesos entre 0 e teto,
    partindo de `w0` (pesos iguais, se não for dado)."""
    if n * teto < 1:               # n pesos <= teto não somam 1
        raise ValueError(f"teto de {teto:.0%} com {n} ativos")
    w0 = np.full(n, 1 / n) if w0 is None else w0
    for ftol in (1e-12, 1e-9):     # relaxa se não convergir
        res = minimize(objetivo, w0,
                       method="SLSQP", bounds=[(0, teto)] * n,
                       constraints=[SOMA_1, *restricoes],
                       options={"ftol": ftol, "maxiter": 500})
        if res.success:
            return res.x
    raise RuntimeError(res.message)


def minima_variancia(cov, teto=1.0):
    return otimizar(lambda w: w @ cov @ w, len(cov), teto=teto)


def maximo_sharpe(mu, cov, rf, teto=1.0):
    w0, falta = np.zeros(len(mu)), 1.0   # maior média viável
    for i in np.argsort(mu)[::-1]:
        w0[i] = min(teto, falta)
        falta -= w0[i]
    if w0 @ mu <= rf:          # sem prêmio a buscar: menor risco
        return minima_variancia(cov, teto)

    def menos_sharpe(w):
        return -(w @ mu - rf) / vol(w, cov)
    return otimizar(menos_sharpe, len(mu), teto=teto, w0=w0)


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
