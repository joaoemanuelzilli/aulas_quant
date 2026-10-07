"""Matriz de correlação das sete ações, 2015-2024."""
import sys; sys.path.insert(0, "..")
from estilo import *
from carteira import retornos

r = retornos("2015", "2024")
c = r.corr().values
nomes = [a[:5] for a in r.columns]
ordem = np.argsort(-c.sum(axis=0))       # mais correlacionadas antes
c = c[np.ix_(ordem, ordem)]
nomes = [nomes[i] for i in ordem]
fig, ax = figura(altura=64)
im = ax.imshow(c, cmap=SEQ_ROXO, vmin=0, vmax=1)
for i in range(len(c)):
    for j in range(len(c)):
        ax.text(j, i, f"{c[i, j]:.2f}".replace(".", ","),
                ha="center", va="center", fontsize=6.8,
                color="white" if c[i, j] > 0.55 else TINTA)
ax.set_xticks(range(len(c)), nomes); ax.set_yticks(range(len(c)), nomes)
ax.tick_params(length=0, labelsize=6.2); ax.grid(False)
for lado in ax.spines.values():
    lado.set_visible(False)
ax.set_aspect("equal")
salvar(fig, "cap06/correlacao")
fora = ~np.eye(len(c), dtype=bool)
print(f"correlação média: {c[fora].mean():.3f}  "
      f"mín {c[fora].min():.3f}  máx {c[fora].max():.3f}")
