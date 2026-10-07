"""Fronteira eficiente das sete ações, 2015-2024."""
import sys; sys.path.insert(0, "..")
from estilo import *
from carteira import (cdi_anual, fronteira, insumos, maximo_sharpe,
                      minima_variancia, retornos, vol)

r = retornos("2015", "2024")
mu, cov = insumos(r)
rf = cdi_anual("2015", "2024")
n = len(mu)
rng = np.random.default_rng(6)
W = rng.dirichlet(np.ones(n), size=4000)       # nuvem aleatória
nuvem_mu = W @ mu
nuvem_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))

w_mv, w_ms = minima_variancia(cov), maximo_sharpe(mu, cov, rf)
alvos = np.linspace(mu.min(), mu.max(), 60)
vols = fronteira(mu, cov, alvos)
efic = alvos >= w_mv @ mu

# fronteira sem restrição de sinal (hipérbole, fórmula fechada)
ic = np.linalg.inv(cov); um = np.ones(n)
A, B, C = um @ ic @ um, um @ ic @ mu, mu @ ic @ mu
m = np.linspace(0.02, 0.46, 200)
hip = np.sqrt((A * m**2 - 2 * B * m + C) / (A * C - B**2))

fig, ax = figura(altura=78)
ax.scatter(nuvem_vol, nuvem_mu, s=1.6, color="#D9C6EF", zorder=1,
           rasterized=True)
ax.plot(hip, m, color=CINZA, lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.plot(vols[~efic], alvos[~efic], color=ROXO, lw=0.9,
        ls=(0, (1, 2)), zorder=3)
ax.plot(vols[efic], alvos[efic], color=ROXO, lw=1.6, zorder=3)
s_ms, m_ms = vol(w_ms, cov), w_ms @ mu
x = np.array([0, 0.56])
ax.plot(x, rf + (m_ms - rf) / s_ms * x, color=OCRE, lw=1.0, zorder=2)
ax.plot(0, rf, "o", color=OCRE, ms=4, clip_on=False, zorder=5)
ax.plot(vol(w_mv, cov), w_mv @ mu, "D", color=TINTA, ms=4, zorder=5)
ax.plot(s_ms, m_ms, "*", color=TINTA, ms=8, zorder=5)
dp = np.sqrt(np.diag(cov))
ax.plot(dp, mu, "o", color=TINTA, ms=3, zorder=4)
for nome, x_, y_ in zip(r.columns.str[:5], dp, mu):
    ax.annotate(nome, (x_, y_), xytext=(5, -1), fontsize=6.8,
                textcoords="offset points", va="center")
nota = dict(textcoords="offset points", fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.annotate("mínima variância", (vol(w_mv, cov), w_mv @ mu),
            xytext=(-44, -20), ha="center", **nota)
ax.annotate("máximo Sharpe", (s_ms, m_ms), xytext=(-58, 16), **nota)
ax.annotate("CDI", (0, rf), xytext=(6, -9), fontsize=7,
            textcoords="offset points")
ax.annotate("linha de alocação\nde capital", (0.5, rf + (m_ms - rf) / s_ms * 0.5),
            xytext=(8, -24), **nota)
ax.annotate("fronteira sem\nrestrição de sinal", (hip[-30], m[-30]),
            xytext=(-78, 6), color=CINZA, **nota)
ax.annotate("fronteira eficiente\n(pesos $\\geq 0$)", (vols[42], alvos[42]),
            xytext=(-84, 12), **nota)
ax.set_xlim(0, 0.75); ax.set_ylim(0, 0.46)
ax.set_xlabel("volatilidade anual"); ax.set_ylabel("retorno esperado anual")
pct(ax, "x"); pct(ax, "y")
salvar(fig, "cap06/fronteira")
print("nuvem: maior Sharpe", ((nuvem_mu - rf) / nuvem_vol).max().round(3),
      "menor vol", nuvem_vol.min().round(4))
print("ótimos: Sharpe", round((m_ms - rf) / s_ms, 3), "vol mín", round(vol(w_mv, cov), 4))
