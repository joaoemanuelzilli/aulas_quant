"""Distribuição da estatística de Dickey-Fuller sob a hipótese nula."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy import stats

rng = np.random.default_rng(1979)
T, N = 500, 50_000


def tau(y, det):
    """Estatística t de gama em  dy_t = det + gama * y_{t-1} + e_t  (uma série por linha)."""
    dy, yl = np.diff(y, axis=1), y[:, :-1]
    n = dy.shape[1]
    cols = [yl]
    if det in ("c", "ct"): cols.append(np.ones_like(yl))
    if det == "ct": cols.append(np.broadcast_to(np.arange(n, dtype=float), yl.shape))
    X = np.stack(cols, axis=2)                       # N x n x k
    XtX = np.einsum("nti,ntj->nij", X, X); Xty = np.einsum("nti,nt->ni", X, dy)
    inv = np.linalg.inv(XtX); beta = np.einsum("nij,nj->ni", inv, Xty)
    res = dy - np.einsum("nti,ni->nt", X, beta)
    s2 = (res ** 2).sum(axis=1) / (n - X.shape[2])
    return beta[:, 0] / np.sqrt(s2 * inv[:, 0, 0])


y = rng.standard_normal((N, T)).cumsum(axis=1)       # passeios aleatórios: H0 verdadeira
taus = {d: tau(y, d) for d in ("n", "c", "ct")}
for d, nome in [("n", "sem constante"), ("c", "com constante"), ("ct", "constante e tendência")]:
    q = np.quantile(taus[d], [0.01, 0.05, 0.10])
    print(f"{nome:22s} quantis 1%, 5%, 10%: {q.round(2)}   P(tau < -1,645) = {np.mean(taus[d] < -1.645):.1%}")
print("média e mediana (com constante):", taus["c"].mean().round(2), np.median(taus["c"]).round(2))

fig, ax = figura(altura=55)
g = np.linspace(-5.2, 3.6, 400)
ax.hist(taus["c"], bins=np.linspace(-5.2, 3.6, 89), density=True, color=ROXO, alpha=0.85)
ax.plot(g, stats.norm.pdf(g), color=CINZA, lw=1.1, ls=(0, (4, 3)))
q5 = np.quantile(taus["c"], 0.05)
ymax = 0.68
ax.plot([q5, q5], [0, 0.53], color=TINTA, lw=0.8)
ax.plot([-1.645, -1.645], [0, 0.53], color=TINTA, lw=0.8, ls=(0, (3, 2)), zorder=5)
ax.text(q5 - 0.08, 0.54, "valor crítico de 5%\nde Dickey-Fuller: " + f"{q5:.2f}".replace(".", ",").replace("-", "−"), fontsize=7, ha="right", va="bottom")
ax.text(-1.645 + 0.08, 0.54, "valor crítico de 5%\nda normal: −1,645", fontsize=7, ha="left", va="bottom", color=CINZA)
ax.text(1.35, 0.30, "normal padrão", fontsize=7, color=CINZA)
ax.text(-4.9, 0.20, "estatística do teste\nsob raiz unitária", fontsize=7)
ax.set_ylim(0, ymax); ax.set_yticks([]); ax.grid(False)
ax.set_xlabel(r"estatística $t$ de $\gamma$ na regressão de Dickey-Fuller com constante")
ax.set_xlim(-5.2, 3.6)
salvar(fig, "cap04/df_dist")
