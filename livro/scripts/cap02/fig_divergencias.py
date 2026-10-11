"""Divergências entre Yahoo e B3 por ano e por tipo."""
import sys; sys.path.insert(0, "..")
from estilo import *
from conferir_fonte import conferir, resumir

papeis = ["ABEV3", "BBAS3", "BOVA11", "ITUB4", "MGLU3", "PETR4", "VALE3", "WEGE3"]
tudo, div = conferir(papeis)
r = resumir(div, "tipo").drop(index="total").drop(columns="total")
tipos = {"fechamento errado": ("fechamento errado", ROXO), "eco": ("eco no pregão seguinte", OCRE),
         "volume zero": ("preço repetido, volume zero", VERDE), "repetido": ("preço repetido, com volume", AZUL)}
fig, ax = figura(altura=56)
base = np.zeros(len(r))
anos = r.index.astype(int)
for col, (nome, cor) in tipos.items():
    ax.bar(anos, r[col], bottom=base, color=cor, width=0.72, label=nome)
    base += r[col].values
for x, v in zip(anos, base):
    ax.text(x, v + 4, f"{int(v)}", ha="center", va="bottom", fontsize=6.6)
ax.set_xticks(anos); ax.set_xticklabels([str(a)[2:] for a in anos])
ax.set_xlabel("ano (20xx)")
ax.set_title("Retornos diários do Yahoo que diferem mais de 0,5 ponto do oficial")
ax.legend(loc="upper right", ncol=1)
ax.set_ylim(0, 310)
salvar(fig, "cap02/divergencias")
print(r.sum().to_dict(), int(base.sum()))
