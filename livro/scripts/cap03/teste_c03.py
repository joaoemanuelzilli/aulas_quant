"""Testes de sanidade das contas centrais do capítulo.

Rodar de dentro de scripts/cap03:  python teste_c03.py"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()
n = len(r)

# 1. Jarque-Bera pela fórmula = scipy
s, k = stats.skew(r), stats.kurtosis(r)
jb = n / 6 * (s ** 2 + k ** 2 / 4)
assert np.isclose(jb, stats.jarque_bera(r).statistic)

# 2. estatística t pela fórmula = scipy
t = r.mean() / (r.std() / np.sqrt(n))
assert np.isclose(t, stats.ttest_1samp(r, 0).statistic)
# ... e = (média anual / vol anual) * raiz(anos)
mu_a, sig_a = 252 * r.mean(), r.std() * np.sqrt(252)
assert np.isclose(t, mu_a / sig_a * np.sqrt(n / 252))

# 3. Ljung-Box pela fórmula = statsmodels
m = 10
rho = acf(r, nlags=m)[1:]
q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, m + 1)))
assert np.isclose(q, acorr_ljungbox(r, lags=[m])["lb_stat"].iloc[0])

# 4. t padronizada tem variância 1; momentos da t
for nu in (3, 5, 10):
    c = np.sqrt(nu / (nu - 2))
    assert np.isclose(stats.t(nu, scale=1 / c).var(), 1)
assert np.isclose(stats.t.stats(6, moments="k"), 6 / (6 - 4))

# 5. excesso de curtose: pandas e scipy concordam quando sem viés
assert np.isclose(r.kurt(), stats.kurtosis(r, bias=False))
assert np.isclose(stats.kurtosis(r, fisher=False), stats.kurtosis(r) + 3)

# 6. correlação = covariância / desvios; w' Sigma w = variância da carteira
at = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"]
x = precos[at].dropna().pct_change().loc["2015":"2024"]
dp = x.std().values
assert np.allclose(x.cov().values / np.outer(dp, dp), x.corr().values)
w = np.full(4, 0.25)
assert np.isclose(w @ x.cov().values @ w, (x @ w).var())
assert np.linalg.eigvalsh(x.cov().values).min() > 0

# 7. Spearman = Pearson dos postos
a, b = x["PETR4.SA"], x["VALE3.SA"]
assert np.isclose(a.corr(b, method="spearman"), a.rank().corr(b.rank()))

# 8. CDI acumulado entre pregões: soma dos logs = log do fator total
bcb = pd.read_csv("../../dados/c02_bcb.csv", index_col=0, parse_dates=True)
ac = np.log1p(bcb["cdi"] / 100).cumsum()
cdi = ac.reindex(ibov.index, method="ffill").diff().dropna()
assert np.isclose(cdi.sum(), ac[ibov.index[-1]] - ac[ibov.index[0]])
assert (cdi > 0).all() and cdi.max() < 0.005

print("todos os testes passaram")
