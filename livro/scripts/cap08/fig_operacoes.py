"""Resultado de cada operação, em ordem crescente: reversão (BOVA11) e rompimento (PETR4)."""
import sys; sys.path.insert(0, "..")
from estilo import *
from motor import carregar, carregar_ohlc, operacoes
from regras import reversao, donchian

p, ret, cdi = carregar("BOVA11.SA")
rev = operacoes(reversao(p["BOVA11.SA"]), ret["BOVA11.SA"], cdi).sort_values()
o, ret, cdi = carregar_ohlc()
don = operacoes(donchian(o, 20), ret, cdi).sort_values()

fig, eixos = figura(altura=62, ncols=2, sharey=True)
for ax, ops, tit in zip(eixos, (rev, don), ("Reversão à média, BOVA11", "Rompimento de canal, PETR4")):
    cores = [ROXO if v > 0 else OCRE for v in ops]
    ax.bar(range(len(ops)), ops.values, color=cores, width=0.8)
    ax.axhline(0, color=TINTA, lw=0.6)
    ax.set_title(tit); ax.set_xlabel(f"{len(ops)} operações, da pior à melhor")
    ax.set_xticks([])
    ax.text(0.04, 0.93, f"{(ops > 0).mean():.0%} com ganho".replace(".", ","), transform=ax.transAxes, fontsize=7, va="top")
pct(eixos[0])
salvar(fig, "cap08/operacoes")
print(rev.describe().round(3).to_dict()); print(don.describe().round(3).to_dict())
