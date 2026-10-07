import numpy as np
from scipy import stats
from scipy.special import xlogy
from carteira import perda, V, Rp

def kupiec(x, T, p):
    """Teste de proporção de falhas: estatística LR e p-valor."""
    def logver(q):      # log-verossimilhança de x falhas em T
        return xlogy(x, q) + xlogy(T - x, 1 - q)
    lr = -2 * (logver(p) - logver(x / T))
    return lr, stats.chi2.sf(lr, df=1)

if __name__ == "__main__":
    jan = perda.rolling(250)
    com_var = jan.mean().shift(1).notna()   # dias com previsão
    ewma = (Rp ** 2).ewm(alpha=0.06, adjust=False).mean()
    for a in (0.95, 0.99):
        z = stats.norm.ppf(a)
        modelos = {
            "histórico 250d": jan.quantile(a),
            "normal 250d": jan.mean() + z * jan.std(),
            "normal EWMA": z * V * np.sqrt(ewma),
        }
        for nome, var in modelos.items():
            var = var.shift(1)      # só dados até a véspera
            viol = (perda > var)[com_var]["2011":"2025"]
            x, T = viol.sum(), len(viol)
            lr, pv = kupiec(x, T, 1 - a)
            print(f"{a:.0%} {nome:14s} {x:3d} em {T} "
                  f"({x / T:.2%})  LR = {lr:5.2f}  p = {pv:.4f}")
