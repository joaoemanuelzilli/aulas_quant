"""Simulação fora da amostra de regras de alocação."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from carteira import (carregar_cdi, insumos, maximo_sharpe,
                      minima_variancia)

JANELA = 756                      # três anos de pregões


def pesos(passado, rf):
    """Pesos de cada regra, estimados só com `passado`."""
    n = passado.shape[1]
    mu, cov = insumos(passado)
    lw = LedoitWolf().fit(passado.values).covariance_ * 252
    inv_vol = 1 / np.sqrt(np.diag(cov))
    return {"1/N": np.full(n, 1 / n),
            "mín. variância": minima_variancia(cov),
            "mín. var. (LW)": minima_variancia(lw),
            "inverso da vol": inv_vol / inv_vol.sum(),
            "máx. Sharpe": maximo_sharpe(mu, cov, rf),
            "máx. Sharpe 30%": maximo_sharpe(mu, cov, rf, 0.30)}


def simular(r, cdi, atraso=0):
    """Retornos diários e giro mensal de cada regra."""
    # `cdi` é fatiado por posição: tem de ter as datas de `r`
    assert cdi.index.equals(r.index), "r e cdi desalinhados"
    diario, giro, antes = {}, {}, {}
    meses = r.index.to_period("M")
    for mes in meses.unique():
        hoje = np.argmax(meses == mes)       # 1º pregão do mês
        if hoje < JANELA + atraso:
            continue
        fim = hoje - atraso                  # não inclui `fim`
        passado = r.iloc[fim - JANELA:fim]
        rf = cdi.iloc[fim - JANELA:fim].mean() * 252
        cotas = (1 + r[meses == mes]).cumprod()
        for nome, w in pesos(passado, rf).items():
            valor = cotas @ w
            ret = valor.pct_change().fillna(valor.iloc[0] - 1)
            diario.setdefault(nome, []).append(ret)
            if nome in antes:
                troca = np.abs(w - antes[nome]).sum() / 2
                giro.setdefault(nome, []).append(troca)
            antes[nome] = w * cotas.iloc[-1] / valor.iloc[-1]
    diario = pd.DataFrame(
        {k: pd.concat(v) for k, v in diario.items()})
    return diario, pd.DataFrame(giro)


def resumo(diario, giro, cdi):
    excesso = diario.sub(cdi.loc[diario.index], axis=0)
    anos = len(diario) / 252
    curva = (1 + diario).cumprod()
    return pd.DataFrame({
        "ret a.a.": (curva.iloc[-1] ** (1 / anos) - 1) * 100,
        "vol a.a.": diario.std() * np.sqrt(252) * 100,
        "Sharpe": excesso.mean() / excesso.std() * np.sqrt(252),
        "pior queda": (curva / curva.cummax() - 1).min() * 100,
        "giro/mês": giro.mean() * 100})
