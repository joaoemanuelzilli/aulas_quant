"""O sinal 20/100 em PETR4 é significativo? Três testes.

1. teste t do excesso médio sobre o CDI (iid e Newey-West);
2. bootstrap em blocos de 21 pregões do Sharpe;
3. placebo: o mesmo sinal deslocado no tempo por todos os
   deslocamentos de 252 pregões a n - 252 (mesmo tempo comprado,
   mesmo giro); sem sorteio, não depende de semente.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import metricas as m
from motor import carregar

JANELA = slice("2011", "2025")
CUSTO = 0.001


def t_media(d, defasagens=None):
    """Estatística t da média; com defasagens, Newey-West."""
    d = np.asarray(d)
    if defasagens is None:
        return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    ajuste = sm.OLS(d, np.ones(len(d))).fit(
        cov_type="HAC", cov_kwds={"maxlags": defasagens})
    return ajuste.tvalues[0]


def bootstrap_blocos(d, bloco, n, rng):
    """Sharpes anualizados de n reamostras em blocos móveis."""
    d = np.asarray(d)
    T = len(d)
    k = int(np.ceil(T / bloco))
    saida = np.empty(n)
    for i in range(n):
        ini = rng.integers(0, T - bloco + 1, k)
        idx = (ini[:, None] + np.arange(bloco)).ravel()[:T]
        x = d[idx]
        saida[i] = x.mean() / x.std(ddof=1) * np.sqrt(252)
    return saida


def retorno_do_sinal(sinal, r_ativo, cdi, custo=CUSTO):
    """Execução no fechamento do sinal, como no motor."""
    posicao = sinal.shift(1).fillna(0.0)
    giro = posicao.diff().abs().fillna(0.0)
    return (posicao * r_ativo + (1 - posicao) * cdi
            - custo * giro), posicao


def placebo(sinal, r_ativo, cdi, folga=252):
    """Sharpes do sinal deslocado circularmente, um para cada
    deslocamento k de folga a n - folga (todos, sem sorteio)."""
    valores = sinal.to_numpy()
    ks = range(folga, len(valores) - folga)
    saida = np.empty(len(ks))
    for i, k in enumerate(ks):
        falso = pd.Series(np.roll(valores, k), sinal.index)
        r, _ = retorno_do_sinal(falso, r_ativo, cdi)
        saida[i] = m.sharpe(r.loc[JANELA], cdi.loc[JANELA])
    return saida


if __name__ == "__main__":
    preco, ret = carregar("PETR4.SA")
    r_ativo = preco.pct_change().fillna(0.0)
    cdi = ret["cdi"]
    rapida = preco.rolling(20).mean()
    lenta = preco.rolling(100).mean()
    sinal = (rapida > lenta).astype(float)
    r, posicao = retorno_do_sinal(sinal, r_ativo, cdi)
    r, rf = r.loc[JANELA], cdi.loc[JANELA]
    d = r - rf
    sr = m.sharpe(r, rf)

    print(f"Sharpe {sr:.2f} em {len(d) / 252:.1f} anos")
    print(f"excesso médio: t = {t_media(d):.2f} (iid)   "
          f"{t_media(d, 10):.2f} (Newey-West, 10 defasagens)")

    rng = np.random.default_rng(10)
    boot = bootstrap_blocos(d, 21, 5000, rng)
    a, b = np.percentile(boot, [2.5, 97.5])
    print(f"bootstrap em blocos de 21: IC 95% [{a:.2f}, {b:.2f}]"
          f"   P(SR <= 0) = {(boot <= 0).mean():.2f}")

    falsos = placebo(sinal, r_ativo, cdi)
    print(f"placebo ({len(falsos)} deslocamentos): média "
          f"{falsos.mean():.2f}   desvio {falsos.std():.2f}")
    print(f"deslocamentos com Sharpe maior que o real: "
          f"{(falsos > sr).mean():.0%}")

    w = posicao.loc[JANELA].mean()
    mistura = w * r_ativo.loc[JANELA] + (1 - w) * rf
    print(f"mistura fixa de {w:.0%} na ação: Sharpe "
          f"{m.sharpe(mistura, rf):.2f}")
