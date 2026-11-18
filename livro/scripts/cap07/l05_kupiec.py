import numpy as np
from scipy import stats
from scipy.special import xlogy
from carteira import perda, V, Rp, var_historico


def kupiec(x, T, p):
    """Proporção de falhas: estatística LR e p-valor."""
    def logver(q):      # log-verossimilhança de x falhas em T
        return xlogy(x, q) + xlogy(T - x, 1 - q)
    lr = -2 * (logver(p) - logver(x / T))
    return lr, stats.chi2.sf(lr, df=1)


def christoffersen(viol):
    """Independência: violar hoje depende de ontem?"""
    v = viol.to_numpy().astype(int)
    n = np.zeros((2, 2))            # n[i, j]: de ontem i a hoje j
    np.add.at(n, (v[:-1], v[1:]), 1)
    p01, p11 = n[0, 1] / n[0].sum(), n[1, 1] / n[1].sum()
    pm = n[:, 1].sum() / n.sum()    # sem distinguir a véspera
    def logver(a, b):
        return (xlogy(n[0, 1], a) + xlogy(n[0, 0], 1 - a)
                + xlogy(n[1, 1], b) + xlogy(n[1, 0], 1 - b))
    lr = -2 * (logver(pm, pm) - logver(p01, p11))
    return lr, stats.chi2.sf(lr, df=1), p01, p11


if __name__ == "__main__":
    jan = perda.rolling(250)
    com_var = var_historico(perda).notna()  # dias com previsão
    ewma = (Rp ** 2).ewm(alpha=0.06, adjust=False).mean()
    ewma = np.sqrt(ewma)            # volatilidade diária
    for a in (0.95, 0.99):
        z = stats.norm.ppf(a)
        zt = stats.t.ppf(a, 5) * np.sqrt(3 / 5)  # t, 5 g.l.
        modelos = {
            "histórico 250d": var_historico(perda, 250, a),
            "normal 250d": (jan.mean() + z * jan.std()).shift(1),
            "normal EWMA": (z * V * ewma).shift(1),
            "t(5) EWMA": (zt * V * ewma).shift(1),
        }
        print(f"{a:.0%}            viol.   taxa  Kupiec  indep."
              "  pior perda/VaR")
        for nome, var in modelos.items():
            dias = com_var & (perda.index >= "2011")
            viol = (perda > var)[dias]
            x, T = viol.sum(), len(viol)
            pk = kupiec(x, T, 1 - a)[1]
            pi = christoffersen(viol)[1]
            pior = (perda / var)[dias].max()
            print(f"{nome:15s} {x:4d}  {x / T:5.2%}  {pk:6.4f}"
                  f"  {pi:6.4f}  {pior:6.1f}")
    v99 = (perda > var_historico(perda))[dias]
    lr, pv, p01, p11 = christoffersen(v99)
    print(f"{len(v99)} dias; histórico 99%: Kupiec LR = "
          f"{kupiec(v99.sum(), len(v99), 0.01)[0]:.2f}, "
          f"independência LR = {lr:.2f}")
    print(f"violação depois de dia sem violação: {p01:.1%}; "
          f"depois de violação: {p11:.1%}")
