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

# 12. pontas da carteira tangente e CDI composto do período
print("tangente: comprado", round(wt[wt > 0].sum(), 3),
      "vendido", round(-wt[wt < 0].sum(), 3))
c = pd.read_csv("../../dados/c06_cdi.csv", index_col=0,
                parse_dates=True)["cdi"].loc["2015":"2024"]
print("CDI 2015-2024: acumulado", round((1 + c).prod() - 1, 4),
      "ao ano civil", round((1 + c).prod() ** 0.1 - 1, 4),
      "base 252", round((1 + c).prod() ** (252 / len(c)) - 1, 4))

# 13. Sharpe da tangente quando o Sharpe verdadeiro é zero
rng = np.random.default_rng(6)
t_dias = len(r)
chol = np.linalg.cholesky(cov / 252)
s2 = []
for _ in range(2000):
    x = rng.standard_normal((t_dias, n)) @ chol.T   # média zero
    m, s = x.mean(axis=0) * 252, np.cov(x.T) * 252
    s2.append(m @ np.linalg.solve(s, m))
s2 = np.array(s2)
print("tangente sob Sharpe verdadeiro zero: S^2 dos dados",
      round(e @ ic @ e, 3), "| N/A", round(n / anos, 3),
      "| média simulada de S^2", round(s2.mean(), 3),
      "| mediana de S", round(np.median(np.sqrt(s2)), 2),
      "| P(S > S dos dados)",
      round((np.sqrt(s2) > np.sqrt(e @ ic @ e)).mean(), 2))

# 14. máximo Sharpe sem venda a descoberto por forma convexa:
# min y'Sy com y'(mu - rf) = 1, y >= 0; pesos = y / soma(y)
res = minimize(lambda y: y @ cov @ y, um / n, method="SLSQP",
               bounds=[(0, None)] * n,
               constraints=[{"type": "eq",
                             "fun": lambda y: y @ e - 1}],
               options={"ftol": 1e-14, "maxiter": 500})
print("máx. Sharpe, forma convexa x razão: diferença",
      np.abs(res.x / res.x.sum()
             - maximo_sharpe(mu, cov, rf)).max())

# 15. diferença de médias PETR4 - WEGE3 e seu erro-padrão
dif = r["PETR4.SA"] - r["WEGE3.SA"]
print("PETR4 - WEGE3: diferença", round(dif.mean() * 252, 3),
      "erro-padrão", round(dif.std() * np.sqrt(252 / anos), 3))

# 16. betas em três janelas e correção de Forbes-Rigobon
from carteira import ACOES, NOMES, retornos

linhas = {}
for rot, (ini, fim) in {"2023-25": ("2023", "2025"),
                        "2015-24": ("2015", "2024"),
                        "Covid": ("2020-01-24", "2020-03-23")
                        }.items():
    y = retornos(ini, fim, ACOES + ["^BVSP"])
    linhas[rot] = y[ACOES].apply(
        lambda z: z.cov(y["^BVSP"])) / y["^BVSP"].var()
print(pd.DataFrame(linhas).set_axis(NOMES).round(2).T.to_string())
k = (retornos("2020-01-24", "2020-03-23", ["^BVSP"]).var()
     / retornos("2023", "2025", ACOES + ["^BVSP"])["^BVSP"].var()
     ).iloc[0]
rho = 0.75
print("Forbes-Rigobon: k =", round(k, 1), "correlação 0,75 ->",
      round(rho / np.sqrt(1 + (k - 1) * (1 - rho**2)), 2))

# 17. correlação semanal com o dólar (PTAX), 2015-2024
ptax = pd.read_csv("../../dados/c02_bcb.csv", index_col=0,
                   parse_dates=True)["ptax_venda"]
sem = pd.concat([pp[ACOES + ["^BVSP"]], ptax], axis=1).loc[
    "2015":"2024"].resample("W-FRI").last().pct_change().dropna()
print("correlação semanal com o dólar:",
      sem.corr()["ptax_venda"].drop("ptax_venda").round(2)
      .to_dict())

# 18. gabarito do exercício de sensibilidade
def troca(a, b):
    return np.abs(a - b).sum() / 2


i_itub, i_wege = NOMES.index("ITUB4"), NOMES.index("WEGE3")
mu2 = mu.copy(); mu2[i_itub] += 0.02
wt2 = ic @ (mu2 - rf); wt2 /= wt2.sum()
print("(a) tangente, ITUB4 +2 p.p.: troca",
      round(troca(wt, wt2), 3),
      "ITUB4", round(wt[i_itub], 3), "->", round(wt2[i_itub], 3))
ms = maximo_sharpe(mu, cov, rf)
print("(b) máx. Sharpe, ITUB4 +2 p.p.: troca",
      round(troca(ms, maximo_sharpe(mu2, cov, rf)), 4))
mu3 = mu.copy(); mu3[i_wege] -= 0.05
ms3 = maximo_sharpe(mu3, cov, rf)
print("(c) máx. Sharpe, WEGE3 -5 p.p.: troca",
      round(troca(ms, ms3), 3), "WEGE3", round(ms[i_wege], 3),
      "->", round(ms3[i_wege], 3))
cov2 = cov.copy()
cov2[i_itub, :] *= 1.02
cov2[:, i_itub] *= 1.02
print("(d) mín. variância, vol de ITUB4 +2%: troca",
      round(troca(minima_variancia(cov),
                  minima_variancia(cov2)), 3))

# 19. estresse de correlação em forma fechada (1/N de hoje)
print("raiz(0,32 x 17,9^2 + 0,68 x 30,3^2) =",
      round(np.sqrt(0.32 * 17.9**2 + 0.68 * 30.3**2), 1))

# 20. correlações entre as sete ações, 2015-2024
cc = r.corr().values
print("correlação: média", round(cc[fora].mean(), 3), "mín",
      round(cc[fora].min(), 3), "máx", round(cc[fora].max(), 3))
