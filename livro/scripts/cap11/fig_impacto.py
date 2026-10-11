"""Custo por ponta contra tamanho da ordem (lei da raiz quadrada) e capacidade."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l02_custo_ordem import custo_pb, V
from l03_custos import equilibrio, resultado

part = np.logspace(-5, np.log10(0.5), 300)
fixo, imp = custo_pb(part * V)
fig, ax = figura(altura=57)
ax.fill_between(part, 0, fixo, color=CINZA_CLARO)
ax.fill_between(part, fixo, fixo + imp, color=LILAS)
ax.plot(part, fixo + imp, color=ROXO)
ax.text(1.3e-5, fixo / 2, "taxas + meio spread", fontsize=6.8, va="center")
ax.text(0.1, 14, "impacto", fontsize=6.8, va="center", ha="center")
for nome, (bruto, giro) in resultado.items():
    eq = equilibrio(bruto, giro)
    x = ((eq - fixo) / imp[-1]) ** 2 * part[-1]
    ax.plot([part[0], x], [eq, eq], color=TINTA, lw=0.6, ls=(0, (4, 3)))
    ax.plot([x, x], [0, eq], color=TINTA, lw=0.6, ls=(0, (4, 3)))
    ax.plot([x], [eq], "o", mfc="white", mec=ROXO, mew=1.1, ms=4.5, zorder=5)
    ax.text(1.3e-5, eq + 1.5, f"equilíbrio do rebalanceamento {nome}: {eq:.1f} pb".replace(".", ","), fontsize=6.8, va="bottom")
    print(nome, eq, x)
ax.set_xscale("log"); ax.set_xlim(1e-5, 0.5); ax.set_ylim(0, 80)
ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]); ax.set_xticklabels(["0,001%", "0,01%", "0,1%", "1%", "10%"]); ax.minorticks_off()
ax.set_xlabel("tamanho da ordem como fração do volume diário (escala log)"); ax.set_ylabel("custo por ponta (pontos-base)")
salvar(fig, "cap11/impacto")
