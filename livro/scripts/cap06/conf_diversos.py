"""Conferências de fórmulas e números do Capítulo 6."""
import numpy as np
from sklearn.covariance import LedoitWolf

from carteira import (cdi_anual, insumos, maximo_sharpe,
                      minima_variancia, retornos, vol)

r = retornos("2015", "2024")
mu, cov = insumos(r)
rf = cdi_anual("2015", "2024")
n, um = len(mu), np.ones(len(mu))
ic = np.linalg.inv(cov)
nomes = [c[:5] for c in r.columns]

# 1. erro-padrão da média anualizada: sigma / raiz(anos)
anos = len(r) / 252
ep = np.sqrt(np.diag(cov) / anos)
print(f"anos na amostra: {anos:.2f}")
for a, m, e in zip(nomes, mu, ep):
    print(f"  {a}: média {m:6.1%}  erro-padrão {e:5.1%}  "
          f"IC95 [{m - 1.96 * e:6.1%}, {m + 1.96 * e:6.1%}]")

# 2. mínima variância global: variância = 1 / (1' S^-1 1)
w = ic @ um / (um @ ic @ um)
print("var MV:", w @ cov @ w, "=", 1 / (um @ ic @ um))
# todos os ativos têm a mesma covariância com a MV
print("cov(ativo, MV):", (cov @ w).round(5))

# 3. tangente: Sharpe^2 = (mu-rf)' S^-1 (mu-rf)
e = mu - rf
wt = ic @ e / (um @ ic @ e)
print("Sharpe tangente:", (wt @ mu - rf) / vol(wt, cov),
      "=", np.sqrt(e @ ic @ e))
print("1'S^-1(mu-rf) > 0:", um @ ic @ e > 0,
      "| retorno da MV > rf:", w @ mu > rf)

# 4. SLSQP sem limites de sinal reproduz as fórmulas fechadas
from scipy.optimize import minimize
soma = {"type": "eq", "fun": lambda x: x.sum() - 1}
res = minimize(lambda x: x @ cov @ x, um / n, method="SLSQP",
               constraints=[soma], options={"ftol": 1e-14})
print("SLSQP x fechada (MV), maior diferença:",
      np.abs(res.x - w).max())

# 5. Black-Litterman: as duas formas da média a posteriori
tau, delta = 0.05, 2.5
pi = delta * cov @ (um / n)
P = np.zeros((1, n)); P[0, 3], P[0, 2] = 1, -1
q = np.array([0.05])
om = P @ (tau * cov) @ P.T
its = np.linalg.inv(tau * cov)
f1 = np.linalg.solve(its + P.T @ np.linalg.inv(om) @ P,
                     its @ pi + P.T @ np.linalg.inv(om) @ q)
f2 = pi + tau * cov @ P.T @ np.linalg.solve(
    P @ (tau * cov) @ P.T + om, q - P @ pi)
print("BL, diferença entre as duas formas:",
      np.abs(f1 - f2).max())

# 6. Ledoit-Wolf: intensidade de encolhimento por janela
for t in [len(r), 756, 126, 21]:
    x = r.values[-t:]
    lw = LedoitWolf().fit(x)
    s = np.cov(x.T)
    print(f"T = {t:4d}: intensidade {lw.shrinkage_:.3f}  "
          f"condição S {np.linalg.cond(s):7.1f}  "
          f"LW {np.linalg.cond(lw.covariance_):6.1f}")

# 7. contribuições de risco (somam a volatilidade da carteira)
for nome, w_ in [("1/N", um / n), ("MV", minima_variancia(cov)),
                 ("MS", maximo_sharpe(mu, cov, rf))]:
    s = vol(w_, cov)
    cr = w_ * (cov @ w_) / s
    print(nome, "vol", round(s, 4), "soma CR", round(cr.sum(), 4),
          "CR %:", (cr / s * 100).round(1))

# 8. MGLU3: quanto o preço ajustado se multiplicou
import pandas as pd
p = pd.read_csv("../../dados/precos_ajustados.csv", index_col=0,
                parse_dates=True)["MGLU3.SA"].dropna()
print("MGLU3, fim de 2015 ao fim de 2020: x",
      round(p.loc[:"2020"].iloc[-1] / p.loc[:"2015"].iloc[-1], 1))
print("MGLU3, fim de 2020 ao fim de 2024: x",
      round(p.loc[:"2024"].iloc[-1] / p.loc[:"2020"].iloc[-1], 3))

# 9. média aritmética x geométrica de PETR4, mesma amostra
x = r["PETR4.SA"]
print("PETR4: aritmética", round(x.mean() * 252, 4),
      "geométrica",
      round((1 + x).prod() ** (252 / len(x)) - 1, 4))

# 10. parcelas da variância de 1/N com as sete ações
fora = ~np.eye(n, dtype=bool)
vbar, cbar = np.diag(cov).mean(), cov[fora].mean()
var7 = vbar / n + (1 - 1 / n) * cbar
print("vbar", round(vbar, 4), "cbar", round(cbar, 4),
      "var 1/N", round(var7, 4), "parcela das covariâncias",
      round((1 - 1 / n) * cbar / var7, 3),
      "cbar/var", round(cbar / var7, 3))
print("pregões na amostra:", len(r), "| CDI rf:", round(rf, 4))

# 11. BOVA11 com retornos simples e com log-retornos; maior
# queda do Ibovespa no conjunto de dados
pp = pd.read_csv("../../dados/precos_ajustados.csv", index_col=0,
                 parse_dates=True)
b = pp["BOVA11.SA"].dropna().loc["2015":"2024"]
print("vol BOVA11 simples",
      round(b.pct_change().std() * np.sqrt(252), 4), "log",
      round(np.log(b).diff().std() * np.sqrt(252), 4))
ib = pp["^BVSP"].dropna()
dd = ib / ib.cummax() - 1
print("Ibovespa, pior queda:", round(dd.min(), 4), "em",
      dd.idxmin().date(), "desde",
      ib.loc[:dd.idxmin()].idxmax().date())
print("vol PETR4 simples:", round(x.std() * np.sqrt(252), 4))
