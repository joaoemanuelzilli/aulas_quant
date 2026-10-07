import numpy as np
import pandas as pd

from carteira import ACOES, retornos, vol

hoje = pd.read_csv("../../dados/c06_carteiras_hoje.csv",
                   index_col=0)
r = retornos("2023", "2025", ACOES + ["^BVSP"])
acoes, ibov = r[ACOES], r["^BVSP"]
cov = acoes.cov().values

# (a) choque no índice, propagado pelos betas
beta = acoes.apply(lambda x: x.cov(ibov)) / ibov.var()
via_beta = beta * -0.20

# (b) choque em um ativo, propagado pela covariância
i = ACOES.index("PETR4.SA")
via_cov = pd.Series(cov[:, i] / cov[i, i] * -0.30, index=ACOES)

ativos = pd.DataFrame({"beta": beta,
                       "Ibov -20% (%)": via_beta * 100,
                       "PETR4 -30% (%)": via_cov * 100})
print(ativos.T.round(2).rename(columns=lambda c: c[:5])
      .to_string())
perdas = pd.DataFrame({"Ibov -20%": via_beta @ hoje,
                       "PETR4 -30%": via_cov @ hoje}) * 100
print("\nperda da carteira em %:")
print(perdas.round(1).to_string())

# (c) estresse de correlação: mistura com a matriz de uns
dp = np.sqrt(np.diag(cov))
corr = cov / np.outer(dp, dp)
fora = ~np.eye(len(dp), dtype=bool)
calma = corr[fora].mean()
crise = retornos("2020-01-24", "2020-03-23").corr().values
crise = crise[fora].mean()
print(f"\ncorrelação média: {calma:.2f} em 2023-2025, "
      f"{crise:.2f} na Covid-19")
lam_crise = (crise - calma) / (1 - calma)
linhas = {}
for lam in [0, lam_crise, 1]:
    c = (1 - lam) * corr + lam * np.ones_like(corr)
    cov_e = c * np.outer(dp, dp) * 252
    linhas[f"λ = {lam:.2f}"] = {
        k: vol(hoje[k].values, cov_e) * 100 for k in hoje}
print("volatilidade anual da carteira em %:")
print(pd.DataFrame(linhas).round(1).to_string())
