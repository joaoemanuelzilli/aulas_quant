import numpy as np
import pandas as pd
import statsmodels.api as sm

px = pd.read_csv("../../dados/precos_ajustados.csv",
                 index_col=0, parse_dates=True)
bcb = pd.read_csv("../../dados/c02_bcb.csv",
                  index_col=0, parse_dates=True)
p = pd.concat([px[["VALE3.SA", "^BVSP", "BOVA11.SA"]],
               bcb["ptax_venda"]], axis=1).dropna()
p.columns = ["vale", "ibov", "bova", "dolar"]
# log-retornos semanais, de sexta a sexta
r = np.log(p.resample("W-FRI").last()).diff().dropna()


def mqo(y, cols, dados=r):
    """MQO com erros-padrão de Newey-West (4 semanas)."""
    X = sm.add_constant(dados[cols])
    return sm.OLS(dados[y], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4})


so_dolar = mqo("vale", ["dolar"])
duas = mqo("vale", ["ibov", "dolar"])
b, t = duas.params, duas.tvalues
print(f"{len(r)} semanas")
print(f"só o dólar:  {so_dolar.params['dolar']:+.3f}"
      f"  (t = {so_dolar.tvalues['dolar']:.1f})")
print(f"com o Ibov:  {b['dolar']:+.3f}  (t = {t['dolar']:.1f})"
      f";  Ibov {b['ibov']:+.3f}  R2 {duas.rsquared:.3f}")

# o coeficiente simples = o múltiplo + Ibov x (Ibov ~ dólar)
delta = mqo("ibov", ["dolar"]).params["dolar"]
soma = b["dolar"] + b["ibov"] * delta
print(f"conferência: {b['dolar']:+.3f} + {b['ibov']:.3f}"
      f" x ({delta:+.3f}) = {soma:+.3f}")

# coeficientes padronizados: beta x desvio(x) / desvio(y)
dp = r.std()
z = b[["ibov", "dolar"]] * dp[["ibov", "dolar"]] / dp["vale"]
print(f"padronizados: Ibov {z['ibov']:.2f}"
      f"  dólar {z['dolar']:.2f}")

comum = sm.OLS(r["vale"],
               sm.add_constant(r[["ibov", "dolar"]])).fit()
print(f"ep do dólar: MQO {comum.bse['dolar']:.3f}"
      f" | HAC {duas.bse['dolar']:.3f}")

# variável binária: 1 a partir de 2020
r["de2020"] = (r.index >= "2020-01-01").astype(float)
d = mqo("vale", ["ibov", "dolar", "de2020"])
print(f"binária 2020+: {d.params['de2020']:+.4f}"
      f"  (t = {d.tvalues['de2020']:.1f})")

# multicolinearidade: o Ibovespa e o fundo que o replica
tres = mqo("vale", ["ibov", "bova", "dolar"])
vif = 1 / (1 - mqo("ibov", ["bova", "dolar"]).rsquared)
vif0 = 1 / (1 - mqo("ibov", ["dolar"]).rsquared)  # sem o BOVA11
f = tres.f_test("ibov = 0, bova = 0").pvalue
print(f"ep do Ibov sem o BOVA11: {duas.bse['ibov']:.2f}")
print(f"Ibov e BOVA11 juntos: {tres.params['ibov']:+.2f}"
      f" (ep {tres.bse['ibov']:.2f}) e"
      f" {tres.params['bova']:+.2f} (ep {tres.bse['bova']:.2f})")
print(f"VIF do Ibov: {vif:.0f} (sem o BOVA11, {vif0:.1f})"
      f"   R2 {tres.rsquared:.3f}")
print(f"teste F conjunto: p = {f:.0e}")
