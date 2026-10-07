"""Quanto dos resultados do capítulo depende de 2020 (números citados
no texto; não é listagem impressa)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, lilliefors

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = np.log(ibov / ibov.shift(1)).dropna()
print("pior e melhor dia:", r.idxmin().date(), f"{r.min():.4f}",
      r.idxmax().date(), f"{r.max():.4f}")
print("em retorno simples:", np.expm1(r.min()), np.expm1(r.max()))
z = (r - r.mean()) / r.std()
print("datas com |z| > 5:", [str(d.date()) for d in z[z.abs() > 5].index])
print("|z|>4 por ano:", z[z.abs() > 4].groupby(z[z.abs() > 4].index.year).size().to_dict())
print("|z|>3: negativos", (z < -3).sum(), "positivos", (z > 3).sum())

# s20 emenda 30/12/2019 a 04/01/2021: um único par de vizinhos falsos
# em mais de 3 700; o efeito nas autocorrelações é desprezível
s20 = r[r.index.year != 2020]
print(f"sem 2020: n={len(s20)} assimetria={s20.skew():.3f} "
      f"exc. curtose={s20.kurt():.3f} JB={stats.jarque_bera(s20)}")
z2 = (s20 - s20.mean()) / s20.std()
print("sem 2020, |z|>3:", (z2.abs() > 3).sum(), "esperado normal",
      round(len(s20) * 2 * stats.norm.sf(3), 2),
      " |z|>4:", (z2.abs() > 4).sum(), "esperado",
      round(len(s20) * 2 * stats.norm.sf(4), 3))
print("t ajustada sem 2020:", stats.t.fit(s20))

m = r.resample("ME").sum()
print("três piores meses:", m.sort_values().head(3).round(4).to_dict())
m2 = m.drop(m.idxmin())
print(f"mensal sem o pior mês: assimetria={m2.skew():.3f} "
      f"exc. curtose={m2.kurt():.3f} JB={stats.jarque_bera(m2)}")

print("Ljung-Box(10) retornos, sem 2020:")
print(acorr_ljungbox(s20, lags=[10]))
print("Ljung-Box(10) retornos², sem 2020:")
print(acorr_ljungbox(s20 ** 2, lags=[10]))
print("autocorrelação de ordem 1: completa", r.autocorr(1),
      "sem 2020", s20.autocorr(1))

# autocorrelação de ordem 1 com erro-padrão robusto (White)
y, x = r.iloc[1:].values, sm.add_constant(r.iloc[:-1].values)
comum = sm.OLS(y, x).fit()
robusto = sm.OLS(y, x).fit(cov_type="HC0")
print(f"regressão r_t em r_(t-1): coef={comum.params[1]:.4f} "
      f"t comum={comum.tvalues[1]:.2f} t robusto={robusto.tvalues[1]:.2f} "
      f"p robusto={robusto.pvalues[1]:.3f}")

print("KS com parâmetros estimados (p inválido):",
      stats.kstest(r, "norm", args=(r.mean(), r.std())))
print("Lilliefors:", lilliefors(r))

# números da tabela descritiva citados no texto (Código 3.1)
at = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "BOVA11.SA"]
pa = precos.loc["2015":"2024", at].dropna()
ra = np.log(pa / pa.shift(1)).dropna()
pior = ra["PETR4.SA"].idxmin()
print("PETR4, pior dia:", pior.date(), f"log {ra['PETR4.SA'].min():.4f}",
      f"simples {np.expm1(ra['PETR4.SA'].min()):.4f}")
print("medianas em %:", (ra.median() * 100).round(4).to_dict())
print("pregões com log-retorno exatamente zero:", (ra == 0).sum().to_dict())
print("média/desvio de PETR4:", ra["PETR4.SA"].std() / ra["PETR4.SA"].mean())

# lacunas: depois de uma linha descartada o retorno cobre mais de um pregão
todas = precos.loc["2015":"2024", at]
falhas = todas.index[todas.isna().any(axis=1) & todas.notna().any(axis=1)]
print("linhas com ausência parcial (descartadas):", len(falhas))
pos = {d: i for i, d in enumerate(precos.index)}
for a in at:
    for d in (ra[a].idxmin(), ra[a].idxmax()):
        ant = pa.index[pa.index.get_loc(d) - 1]
        print(a, d.date(), "linhas do arquivo desde o preço anterior:",
              pos[d] - pos[ant])
c = ra["PETR4.SA"].corr(ra["VALE3.SA"], method="spearman")
print("Spearman PETR4 x VALE3 (logs):", c)
