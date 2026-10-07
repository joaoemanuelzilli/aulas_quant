"""ACF dos retornos do Ibovespa e dos retornos ao quadrado."""
import sys; sys.path.insert(0, "..")
from estilo import *
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.stattools import acf

r = 100 * np.log(precos("^BVSP")["^BVSP"].dropna()).diff().dropna()
T, K = len(r), 40
banda = 1.96 / np.sqrt(T)
a1, a2 = acf(r, nlags=K)[1:], acf(r ** 2, nlags=K)[1:]
print(f"T = {T}  banda = {banda:.4f}")
print("ACF retornos k=1..5:", a1[:5].round(3), " fora da banda:", int((abs(a1) > banda).sum()), "de", K)
print("ACF quadrados k=1..5:", a2[:5].round(3), " k=20:", a2[19].round(3), " k=40:", a2[39].round(3),
      " fora da banda:", int((abs(a2) > banda).sum()))
print("maior |ACF| dos retornos: defasagem", int(abs(a1).argmax()) + 1, a1[abs(a1).argmax()].round(3),
      "| primeira defasagem dos quadrados dentro da banda:", int(np.argmax(abs(a2) < banda)) + 1)
aa = acf(abs(r), nlags=K)[1:]
print("ACF |r| k=1, 20, 40:", aa[[0, 19, 39]].round(3))

# t de rho_1 com erro-padrão usual e robusto a heterocedasticidade (White)
y, X = r.values[1:], sm.add_constant(r.values[:-1])
print("t de rho_1: usual", sm.OLS(y, X).fit().tvalues[1].round(2), " robusto (HC0)", sm.OLS(y, X).fit(cov_type="HC0").tvalues[1].round(2))

# quantas vezes um GARCH(1,1) sem autocorrelação nenhuma fura a banda na defasagem 1?
res = arch_model(r, vol="GARCH", p=1, q=1).fit(disp="off")
w, al, be = res.params[["omega", "alpha[1]", "beta[1]"]].values
rng = np.random.default_rng(2026)
N, B = 2000, 500                                   # séries e período de aquecimento
z = rng.standard_normal((N, T + B))
s2 = np.full(N, w / (1 - al - be)); sim = np.empty((N, T + B))
for t in range(T + B):
    sim[:, t] = np.sqrt(s2) * z[:, t]
    s2 = w + al * sim[:, t] ** 2 + be * s2
sim = sim[:, B:]
sim = sim - sim.mean(axis=1, keepdims=True)
rho1 = (sim[:, 1:] * sim[:, :-1]).sum(axis=1) / (sim ** 2).sum(axis=1)
fura = abs(rho1) > banda
print(f"GARCH simulado: |rho_1| fora da banda em {np.mean(fura):.1%} das 2000 séries (nominal: 5%)")

fig, (a, b) = figura(altura=74, nrows=2, sharex=True)
k = np.arange(1, K + 1)
for ax, v, tit in ((a, a1, "ACF dos log-retornos"), (b, a2, "ACF dos log-retornos ao quadrado")):
    ax.axhspan(-banda, banda, color=CINZA_CLARO, zorder=0)
    ax.bar(k, v, width=0.6, color=ROXO)
    ax.axhline(0, color=CINZA, lw=0.6)
    ax.set_title(tit); virgula(ax, casas=1)
a.set_ylim(-0.12, 0.12); a.set_yticks([-0.1, 0, 0.1])
b.set_ylim(-0.05, 0.6)
b.set_xlabel("defasagem $k$ (dias úteis)"); b.set_xlim(0.2, K + 0.8)
salvar(fig, "cap04/acf_ibov")
