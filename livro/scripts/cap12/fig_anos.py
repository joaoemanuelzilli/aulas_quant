"""Retorno ativo por ano: momentum menos pesos iguais."""
import sys; sys.path.insert(0, "..")
from estilo import *
from projeto import *

px, ret, cdi = carregar()
ano = lambda s: (1 + s).groupby(s.index.year).prod() - 1
partes = []
for per in (TREINO, TESTE):
    r, _, _ = simular(alvo_momentum(px, 12, 2), ret, cdi, per)
    b, _, _ = simular(alvo_fixo(px, ACOES), ret, cdi, per)
    partes.append(ano(r) - ano(b))
ativo = pd.concat(partes)
print((ativo * 100).round(1).to_string())
print("anos positivos:", (ativo > 0).sum(), "de", len(ativo))

fig, ax = figura(altura=45)
cores = [CINZA if a < 2020 else ROXO for a in ativo.index]
ax.bar(ativo.index, ativo, color=cores, width=0.68)
ax.axhline(0, color=TINTA, lw=0.6)
for a, v in ativo.items():
    ax.text(a, v + (0.025 if v >= 0 else -0.025),
            f"{v * 100:+.0f}".replace("-", "−"), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=6.8)
ax.set_xticks(ativo.index)
ax.set_ylim(-0.55, 1.32)
ax.set_yticks([-0.4, 0, 0.4, 0.8, 1.2])
pct(ax)
ax.axvline(2019.5, color=TINTA, lw=0.8)
ax.text(2019.35, 1.3, "treino", ha="right", va="top", fontsize=7.2)
ax.text(2019.65, 1.3, "teste", ha="left", va="top", fontsize=7.2)
ax.set_ylabel("retorno ativo no ano")
salvar(fig, "cap12/anos")
