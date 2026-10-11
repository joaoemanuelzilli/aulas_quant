"""Contas de apoio citadas no texto do Capítulo 4 (não impressas como listagem)."""
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

precos = pd.read_csv("../../dados/precos_ajustados.csv", index_col=0, parse_dates=True)
r = 100 * np.log(precos["^BVSP"].dropna()).diff().dropna()

# 1. GARCH(1,1) normal: curtose implícita, incerteza da persistência, subamostras
res = arch_model(r, vol="GARCH", p=1, q=1).fit(disp="off")
w, a, b = res.params[["omega", "alpha[1]", "beta[1]"]]
p = a + b
curt = 3 * (1 - p**2) / (1 - p**2 - 2 * a**2)
print(f"curtose implícita do GARCH normal: {curt:.2f}  (condição (a+b)^2 + 2a^2 = {p**2 + 2 * a**2:.4f} < 1)")
print(f"curtose amostral dos retornos: {((r - r.mean())**4).mean() / r.var(ddof=0)**2:.2f}")
V = res.param_cov.loc[["alpha[1]", "beta[1]"], ["alpha[1]", "beta[1]"]].to_numpy()
ep = np.sqrt(V.sum())
lo, hi = p - 1.96 * ep, p + 1.96 * ep
mv = lambda x: np.log(0.5) / np.log(x)
print(f"persistência {p:.4f}  erro-padrão robusto {ep:.4f}  IC 95%: {lo:.3f} a {hi:.3f}  meia-vida: {mv(lo):.0f} a {mv(hi):.0f} pregões")
for ini, fim in (("2010", "2019"), ("2021", "2025")):
    m = arch_model(r[ini:fim], vol="GARCH", p=1, q=1).fit(disp="off")
    pp = m.params["alpha[1]"] + m.params["beta[1]"]
    print(f"  {ini}-{fim}: persistência {pp:.3f}  meia-vida {mv(pp):.0f}  vol. de longo prazo {np.sqrt(252 * m.params['omega'] / (1 - pp)):.1f}% a.a.")
print(f"meia-vida com persistência 0,98: {mv(0.98):.1f}; com 0,90: {mv(0.90):.1f}; EWMA 0,94: {mv(0.94):.1f}")
rho1 = a * (1 - a * b - b**2) / (1 - 2 * a * b - b**2)
print(f"autocorrelação de ordem 1 dos quadrados implícita no GARCH: {rho1:.3f}")
g = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off").params
print(f"GJR t: persistência alfa + gama/2 + beta = {g['alpha[1]'] + g['gamma[1]'] / 2 + g['beta[1]']:.3f}")

# 2. exemplo "um passo da recursão" com parâmetros redondos e tamanho de posição (limite de VaR de R$ 20 mil a 95%)
v1 = 0.05 + 0.075 * 4**2 + 0.90 * 1.2**2
s1 = np.sqrt(v1)
print(f"exemplo: parcelas {0.05:.4f} + {0.075 * 16:.4f} + {0.90 * 1.44:.4f} = {v1:.4f}; sigma = {s1:.4f}% ao dia = {s1 * np.sqrt(252):.1f}% ao ano; antes {1.2 * np.sqrt(252):.1f}%; alta de {s1 / 1.2 - 1:.0%}")
print(f"variância de longo prazo do exemplo: {0.05 / 0.025:.2f} (vol. {np.sqrt(2):.2f}% ao dia); pesos 0,025 x 2 = {0.025 * 2}")
for s in (1.2, s1):
    print(f"vol. {s:.3f}% ao dia: posição = {20000 / (1.645 * s / 100):,.0f}")
print(f"redução: {1 - 1.2 / s1:.1%}")

# 3. previsão da média fora da amostra com outras datas de corte
def compara(treino, teste, ordem=(1, 0, 0)):
    m = ARIMA(treino.to_numpy(), order=ordem).fit()
    prev = m.append(teste.to_numpy(), refit=False).predict(start=len(treino))
    d = (teste.to_numpy() - prev) ** 2 - (teste.to_numpy() - treino.mean()) ** 2
    return m.params[1], prev, pd.Series(d, index=teste.index)

print("corte   phi      RMSE AR(1)  RMSE média   t")
for corte in ("2015", "2017", "2019", "2021"):
    tr, te = r[:corte], r[str(int(corte) + 1):]
    phi, prev, d = compara(tr, te)
    rm = np.sqrt(np.mean((te.to_numpy() - prev) ** 2)); rn = np.sqrt(np.mean((te.to_numpy() - tr.mean()) ** 2))
    print(f"{corte}  {phi:7.4f}  {rm:.4f}      {rn:.4f}    {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):5.2f}")
for ordem in ((1, 0, 0), (2, 0, 2)):
    _, _, d = compara(r[:"2021"], r["2022":], ordem)
    por_ano = d.groupby(d.index.year).mean()
    print(f"corte 2021, ARIMA{ordem}: diferença média de erro² por ano:", por_ano.round(3).to_dict())

# 4. previsão da variância fora da amostra (parâmetros fixos em 2021)
tr, te = r[:"2021"], r["2022":]
mg = arch_model(tr, vol="GARCH", p=1, q=1).fit(disp="off")
wg, ag, bg = mg.params[["omega", "alpha[1]", "beta[1]"]]
mu = mg.params["mu"]
e = (r - mu).to_numpy()
s2 = np.empty(len(r)); s2[0] = tr.var()
ew = np.empty(len(r)); ew[0] = tr.var()
for t in range(1, len(r)):
    s2[t] = wg + ag * e[t - 1] ** 2 + bg * s2[t - 1]
    ew[t] = 0.94 * ew[t - 1] + 0.06 * e[t - 1] ** 2
n = len(tr)
alvo = (te.to_numpy() - tr.mean()) ** 2
eq = {"variância do treino": np.mean((alvo - tr.var()) ** 2), "GARCH(1,1)": np.mean((alvo - s2[n:]) ** 2), "EWMA 0,94": np.mean((alvo - ew[n:]) ** 2)}
for k, v in eq.items():
    print(f"erro quadrático da previsão de variância, {k}: {v:.2f}  ({v / eq['variância do treino'] - 1:+.0%})")
print(f"variância realizada média no teste {alvo.mean():.2f}; previsão média do GARCH {s2[n:].mean():.2f}; do EWMA {ew[n:].mean():.2f}")
