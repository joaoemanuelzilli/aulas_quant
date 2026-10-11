"""Confere as contas feitas à mão no texto do capítulo."""
import numpy as np
from scipy import stats

# Exemplo: dois ativos, R$ 100 mil
V = 100_000
w = np.array([0.6, 0.4]); sig = np.array([0.02, 0.03]); rho = 0.3
S = np.outer(sig, sig) * np.array([[1, rho], [rho, 1]])
sp = np.sqrt(w @ S @ w)
z95, z99 = stats.norm.ppf([0.95, 0.99])
print("variância:", w @ S @ w, " sigma_p:", sp)
print("z95, z99:", z95, z99)
print("VaR95 carteira:", z95 * sp * V, " VaR99:", z99 * sp * V)
print("VaR95 isolados:", z95 * sig * w * V, (z95 * sig * w * V).sum())
print("10 dias:", z95 * sp * V * np.sqrt(10), np.sqrt(10))

# quantis e ES padronizados
for a in (0.95, 0.975, 0.99):
    z = stats.norm.ppf(a)
    print(f"normal {a}: z={z:.4f} ES={stats.norm.pdf(z) / (1 - a):.4f}")
for nu in (3, 4, 5, 10):
    k = np.sqrt((nu - 2) / nu)
    for a in (0.95, 0.99):
        q = stats.t.ppf(a, nu)
        es = stats.t.pdf(q, nu) / (1 - a) * (nu + q * q) / (nu - 1)
        print(f"t nu={nu} {a}: q={q:.4f} fator={k:.4f} "
              f"q*fator={q * k:.4f} ES*fator={es * k:.4f} ES/VaR={es / q:.3f}")
# confere a fórmula do ES da t por integração numérica
from scipy import integrate
nu, a = 4.49, 0.99
q = stats.t.ppf(a, nu)
num = integrate.quad(lambda x: x * stats.t.pdf(x, nu), q, np.inf)[0] / (1 - a)
print("ES t fórmula x integral:", stats.t.pdf(q, nu) / (1 - a) * (nu + q * q) / (nu - 1), num)

# subaditividade: dois títulos independentes, calote 4%, perda 100
p = 0.04
print("P(pelo menos um calote):", 1 - (1 - p) ** 2, " P(dois):", p * p)
rng = np.random.default_rng(0)
d = rng.random((2_000_000, 2)) < p
L1, L12 = 100 * d[:, 0], 100 * d.sum(axis=1)
def es(L, a):
    s = np.sort(L)[::-1]; k = int(round(len(L) * (1 - a)))
    return s[:k].mean()
print("VaR95 um:", np.quantile(L1, 0.95, method="inverted_cdf"),
      " VaR95 dois:", np.quantile(L12, 0.95, method="inverted_cdf"))
print("ES95 um (exato 80):", es(L1, 0.95), " dois (exato 103,2):", es(L12, 0.95))
print("ES exato dois:", (p * p * 200 + (0.05 - p * p) * 100) / 0.05)

# Kupiec: exemplo à mão, 250 dias, 99% (inclui x = 0)
from l05_kupiec import kupiec
for x in range(0, 11):
    lr, pv = kupiec(x, 250, 0.01)
    print(x, round(lr, 3), round(pv, 4))
# exercício: 36 violações em 500 dias a 95%
print("exercício Kupiec:", kupiec(36, 500, 0.05), [x for x in range(1, 60) if kupiec(x, 500, 0.05)[0] > 3.841])
print("crítico 5%:", stats.chi2.ppf(0.95, 1))
print("P(0 violações em 250):", 0.99 ** 250)
# erro-padrão do quantil normal: sqrt(a(1-a)/n)/phi(z), em unidades de sigma
for n in (250, 2500):
    for a in (0.95, 0.99):
        z = stats.norm.ppf(a)
        print(n, a, "obs. além:", n * (1 - a), "ep/quantil:", np.sqrt(a * (1 - a) / n) / stats.norm.pdf(z) / z)
# recuperação de um drawdown
for d_ in (0.1, 0.2, 0.429, 0.5):
    print(d_, 1 / (1 - d_) - 1)

# ---- quarta rodada
# tamanho exato e poder do teste de Kupiec com 250 dias a 99%
rej = [x for x in range(0, 251) if kupiec(x, 250, 0.01)[0] > 3.841]
print("região de rejeição (250 dias):", rej[:3], "...")
b = stats.binom(250, 0.01)
print("P(0) =", b.pmf(0), " P(>=7) =", b.sf(6), " tamanho exato =", b.pmf(0) + b.sf(6))
b2 = stats.binom(250, 0.02)
print("poder contra 2%:", sum(b2.pmf(x) for x in rej))
# exemplo de VaR e ES em vinte dias
x = np.array(list(range(1, 17)) + [18, 22, 30, 50.0])
print("vinte dias: VaR 90% =", np.quantile(x, 0.9), " ES 90% =", x[x >= np.quantile(x, 0.9)].mean())
x[-1] = 500
print("com 500:    VaR 90% =", np.quantile(x, 0.9), " ES 90% =", x[x >= np.quantile(x, 0.9)].mean())
# receita de dois ativos correlacionados
rho = 0.6
print("coeficiente:", np.sqrt(1 - rho ** 2), " variância:", rho ** 2 + (1 - rho ** 2))
z = rng.standard_normal((1_000_000, 2))
x2 = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
print("correlação simulada:", np.corrcoef(z[:, 0], x2)[0, 1], np.linalg.cholesky([[1, rho], [rho, 1]]))
# tempo de zeragem
print("pregões para zerar:", 50 / (0.2 * 20), " raiz:", np.sqrt(50 / (0.2 * 20)))
# VaR em % do patrimônio; erro-padrão do quantil no Monte Carlo com mil cenários
from carteira import perda, V
L = perda["2015":"2024"]
print("VaR % do patrimônio:", np.quantile(L, [0.95, 0.99]) / V)
print("ep relativo do quantil 99%, N = 1000:", np.sqrt(0.99 * 0.01 / 1000) / stats.norm.pdf(z99) / z99)
# recuperação do drawdown ao CDI médio de 2015 a 2024
import pandas as pd
cdi = pd.read_csv("../../dados/c02_bcb.csv", index_col=0, parse_dates=True)["cdi"].dropna()["2015":"2024"] / 100
aa = (1 + cdi).prod() ** (1 / 10) - 1      # dez anos de calendário
print("CDI a.a. 2015-2024:", aa, " cota no vale:", 1 - 0.429, " anos para +75,1%:", np.log(1 / (1 - 0.429)) / np.log(1 + aa))
# exercício 1
s = 500_000 * 0.012
print("ex.1:", z95 * s, z99 * s, z99 * s * np.sqrt(10), s * stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025)
