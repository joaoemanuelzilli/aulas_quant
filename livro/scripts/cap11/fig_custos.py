"""Sharpe contra custo por ponta, nas duas frequências de rebalanceamento."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l03_custos import resultado, sharpe

custos = np.arange(0, 71, 1)
fig, ax = figura(altura=57)
ax.axhline(0, color=TINTA, lw=0.6)
for (nome, (bruto, giro)), cor in zip(resultado.items(), [ROXO, OCRE]):
    s = [sharpe(bruto - c / 1e4 * giro) for c in custos]
    ax.plot(custos, s, color=cor)
    pts = [0, 5, 10, 20]
    ax.plot(pts, [s[c] for c in pts], "o", color=cor, ms=3.2)
    eq = bruto.sum() / giro.sum() * 1e4
    print(nome, "equilíbrio", round(eq, 1), "sharpe em 60", round(s[-1], 2))
    if eq < 60:
        ax.plot([eq], [0], "o", mfc="white", mec=cor, mew=1.1, ms=4.5, zorder=5)
        ax.annotate(f"equilíbrio\n{eq:.1f} pb".replace(".", ","), (eq, 0), xytext=(eq + (-5.5 if nome == "diário" else 0), -0.1 if nome == "mensal" else -0.12),
                    fontsize=6.8, ha="center", va="center")
    y = s[25] if nome == "diário" else s[25]
    ax.text(25.8, y + (0.035 if nome == "mensal" else 0.03), f"rebalanceamento {nome}", fontsize=7.2, ha="left", va="bottom" if nome == "mensal" else "bottom", rotation=0)
ax.set_xlabel("custo por ponta (pontos-base do valor negociado)"); ax.set_ylabel("índice de Sharpe líquido")
ax.set_xlim(0, 70); ax.set_ylim(-0.55, 0.25)
virgula(ax, "y", 1)
salvar(fig, "cap11/sharpe_custo")
