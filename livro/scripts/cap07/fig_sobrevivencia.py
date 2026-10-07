"""Viés de sobrevivência em um universo simulado de 500 ações."""
import sys; sys.path.insert(0, "..")
from estilo import *
from sobrevivencia import simular, ANOS, DIAS, BARREIRA

valor, morta = simular()
t = np.arange(valor.shape[0]) / DIAS
fig, (a1, a2) = figura(altura=58, ncols=2)
rng = np.random.default_rng(3)
amostra = rng.choice(valor.shape[1], 70, replace=False)
for j in amostra:
    if morta[j]:
        fim = np.argmax(valor[:, j] < BARREIRA)
        a1.plot(t[:fim + 1:5], valor[:fim + 1:5, j], color=OCRE, lw=0.45, alpha=0.9)
        a1.plot([t[fim]], [valor[fim, j]], "o", color=OCRE, ms=2.2)
    else:
        a1.plot(t[::5], valor[::5, j], color=CINZA_CLARO, lw=0.4, zorder=1)
a1.axhline(BARREIRA, color=TINTA, lw=0.6, ls=(0, (4, 3)))
a1.text(0.2, BARREIRA * 0.8, "deslistagem", fontsize=6.8, va="top")
a1.set_yscale("log"); a1.set_ylim(0.08, 150)
a1.set_yticks([0.1, 1, 10, 100]); a1.set_yticklabels(["0,1", "1", "10", "100"])
a1.minorticks_off()
a1.set_title("70 das 500 trajetórias", fontsize=7.6)
a1.set_xlabel("anos"); a1.set_ylabel("valor de R$ 1 (escala log)")

todas = valor.mean(axis=1)
vivas = valor[:, ~morta].mean(axis=1)
a2.plot(t, vivas, color=OCRE); a2.plot(t, todas, color=ROXO)
rotulo_final(a2, t[-1], vivas[-1], "só sobre-\nviventes", OCRE)
rotulo_final(a2, t[-1], todas[-1], "universo\ncompleto", ROXO)
a2.set_title("Carteira de pesos iguais", fontsize=7.6)
a2.set_xlabel("anos"); a2.set_xlim(0, ANOS + 5.2)
a2.spines["bottom"].set_bounds(0, ANOS)
a2.set_xticks([0, 5, 10, 15]); a1.set_xticks([0, 5, 10, 15])
virgula(a2, casas=0)
salvar(fig, "cap07/sobrevivencia")
