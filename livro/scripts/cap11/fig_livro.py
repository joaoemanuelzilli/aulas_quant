"""Livro de ofertas esquemático: quantidade por nível de preço."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l01_livro import compras, vendas

fig, ax = figura(altura=56)
for (pr, q) in compras:
    ax.barh(pr, -q, height=0.0072, color=ROXO)
    ax.text(-q - 60, pr, f"{q:,}".replace(",", "."), ha="right", va="center", fontsize=6.8)
for (pr, q) in vendas:
    ax.barh(pr, q, height=0.0072, color=OCRE)
    ax.text(q + 60, pr, f"{q:,}".replace(",", "."), ha="left", va="center", fontsize=6.8)
bid, ask = compras[0][0], vendas[0][0]
ax.axhline((bid + ask) / 2, color=CINZA, lw=0.6, ls=(0, (4, 3)))
ax.text(3900, 20.0108, "preço médio 20,01", ha="right", va="bottom", fontsize=6.8, color=CINZA)
ax.annotate("", (-1900, ask), (-1900, bid), arrowprops=dict(arrowstyle="<->", lw=0.6, color=TINTA, shrinkA=0, shrinkB=0))
ax.plot([-2000, 0], [ask, ask], color=CINZA, lw=0.4); 
ax.text(-2080, 20.0152, "spread 0,02", ha="right", va="center", fontsize=6.8)
ax.annotate("melhor compra (bid)", (0, 20.00), xytext=(1900, 20.002), fontsize=7, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA, shrinkB=4))
ax.annotate("melhor venda (ask)", (0, 20.0235), xytext=(-1700, 20.043), fontsize=7, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA, shrinkB=4))
ax.text(-3900, 20.00, "ofertas de compra", fontsize=7.2, ha="left", va="center", weight="semibold")
ax.text(3900, 20.03, "ofertas de venda", fontsize=7.2, ha="right", va="center", weight="semibold")
ax.set_yticks([p for p, _ in compras + vendas] + [20.01])
ax.set_yticklabels([f"{p:.2f}".replace(".", ",") for p, _ in compras + vendas] + [""])
ax.set_xlim(-4000, 4000); ax.set_ylim(19.952, 20.068)
ax.set_xticks([-3000, -2000, -1000, 0, 1000, 2000, 3000]); ax.set_xticklabels(["3.000", "2.000", "1.000", "0", "1.000", "2.000", "3.000"])
ax.set_xlabel("quantidade ofertada em cada preço (ações)"); ax.set_ylabel("preço (R$)")
ax.grid(False); ax.axvline(0, color=CINZA, lw=0.6)
salvar(fig, "cap11/livro_ofertas")
