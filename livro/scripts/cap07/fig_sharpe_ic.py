"""Sharpe estimado contra o limiar de significância, por anos de histórico."""
import sys; sys.path.insert(0, "..")
from estilo import *

anos = np.linspace(1, 40, 400)
fig, ax = figura(altura=60)
sr = 0.5
ep = np.sqrt((1 + sr ** 2 / 504) / anos)
ax.fill_between(anos, sr - 1.96 * ep, sr + 1.96 * ep, color=LILAS, lw=0)
ax.plot(anos, sr - 1.96 * ep, color=ROXO, lw=1)
ax.plot(anos, sr + 1.96 * ep, color=ROXO, lw=1)
ax.axhline(sr, color=ROXO, lw=0.8, ls=(0, (4, 3)))
ax.axhline(0, color=TINTA, lw=0.6)
limiar = 1.96 / np.sqrt(anos)                 # estimativa mínima para t = 1,96
ax.plot(anos, limiar, color=OCRE, lw=1.2)
cruza = (1.96 / sr) ** 2 * (1 + sr ** 2 / 504)
ax.plot([cruza], [sr], "o", color=OCRE, ms=4, zorder=5)
ax.annotate(f"{cruza:.1f} anos".replace(".", ",") + ": metade dos\nhistóricos passa do limiar",
            (cruza, sr), xytext=(10, 30), textcoords="offset points", fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.text(39.7, 0.56, "Sharpe verdadeiro = 0,5", ha="right", fontsize=7)
ax.text(2.6, -1.25, "faixa de 95% do\nSharpe estimado", fontsize=7)
ax.annotate("limiar de significância\n$1{,}96/\\sqrt{n}$", (3.0, 1.96 / np.sqrt(3.0)), xytext=(14, 22),
            textcoords="offset points", fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.set_xlabel("anos de histórico diário ($n$)"); ax.set_ylabel("Sharpe anualizado")
ax.set_xlim(1, 40); ax.set_ylim(-1.6, 2.6)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
virgula(ax)
salvar(fig, "cap07/sharpe_ic")
print("cruza em", cruza)
