"""Leque de trajetórias de um movimento browniano geométrico para a capa."""
import sys; sys.path.insert(0, ".")
from estilo import *

rng = np.random.default_rng(2026)
n, T = 260, 252
z = rng.standard_normal((n, T))
mu, sig = 0.10, 0.28
caminhos = np.exp(np.cumsum((mu - sig**2 / 2) / 252 + sig / np.sqrt(252) * z, axis=1))
caminhos = np.hstack([np.ones((n, 1)), caminhos])
fig = plt.figure(figsize=(210 * MM, 150 * MM)); fig.set_constrained_layout(False)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); fig.patch.set_alpha(0)
ordem = np.argsort(caminhos[:, -1])
for i in ordem:
    ax.plot(caminhos[i], color="#7B2FBE", lw=0.45, alpha=0.33)
for q, a in [(5, 0.9), (50, 1), (95, 0.9)]:
    ax.plot(np.percentile(caminhos, q, axis=0), color="white", lw=0.9 if q != 50 else 1.3, alpha=a)
ax.set_xlim(0, T); ax.set_ylim(0.45, 2.05)
FIGURAS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURAS / "capa_leque.pdf", transparent=True); print("ok")
