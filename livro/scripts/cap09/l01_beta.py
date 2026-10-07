import pandas as pd
import statsmodels.api as sm

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
p = precos[["PETR4.SA", "^BVSP"]].dropna()
R = p.pct_change().dropna().loc["2019":"2024"]

X = sm.add_constant(R["^BVSP"])     # coluna de uns: o intercepto
mqo = sm.OLS(R["PETR4.SA"], X).fit()
for tabela in mqo.summary(slim=True).tables:
    print(tabela)

# mesmos coeficientes, erros-padrão de Newey-West (HAC)
hac = sm.OLS(R["PETR4.SA"], X).fit(
    cov_type="HAC", cov_kwds={"maxlags": 5})
print(f"erro-padrão do beta: MQO {mqo.bse.iloc[1]:.4f}"
      f" | HAC {hac.bse.iloc[1]:.4f}")
