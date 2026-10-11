"""Três desfechos da tripla barreira em trechos de PETR4."""
import sys; sys.path.insert(0, "..")
from estilo import *
from features import carregar
from l04_barreiras import tripla_barreira

d = carregar()
p = d["Close"]
sigma = np.log(p).diff().rolling(21).std()
b = tripla_barreira(p, sigma, h=10, b=1.0)
casos = [("2023-05-02", "alvo tocado: rótulo $+1$"),
         ("2022-04-20", "stop tocado: rótulo $-1$"),
         ("2024-06-11", "tempo esgotado: rótulo $0$")]

fig, eixos = figura(altura=56, ncols=3, sharey=True)
for ax, (dia, titulo) in zip(eixos, casos):
    t = p.index.get_loc(dia)
    larg = np.sqrt(10) * sigma.iloc[t]
    caminho = np.exp(np.log(p.iloc[t:t + 11].to_numpy())
                     - np.log(p.iloc[t])) - 1
    n = int(b["dias"].iloc[t])
    cima, baixo = np.exp(larg) - 1, np.exp(-larg) - 1
    ax.plot([0, 10], [cima, cima], color=TINTA, lw=0.8)
    ax.plot([0, 10], [baixo, baixo], color=TINTA, lw=0.8)
    ax.plot([10, 10], [baixo, cima], color=TINTA, lw=0.8)
    ax.axhline(0, color=CINZA_CLARO, lw=0.5, zorder=0)
    ax.plot(range(11), caminho, color=CINZA, lw=0.8,
            ls=(0, (2, 2)))
    ax.plot(range(n + 1), caminho[:n + 1], color=ROXO, lw=1.3)
    ax.plot([n], [caminho[n]], "o", color=ROXO, ms=4)
    ax.set_title(titulo, fontsize=7.4)
    ax.set_xlabel(f"dias após {pd.Timestamp(dia):%d/%m/%Y}",
                  fontsize=7)
    ax.set_xticks([0, 5, 10]); ax.set_xlim(0, 10.8)
    ax.grid(False)
    pct(ax)
    print(dia, int(b["rotulo"].iloc[t]), n,
          f"largura {np.exp(larg) - 1:.2%}",
          f"final 10d {caminho[-1]:.2%}")
eixos[0].set_ylabel("retorno desde a entrada")
kw = dict(fontsize=6.6, color=TINTA)
eixos[2].text(0.3, 0.071, "barreira de lucro", va="bottom", **kw)
eixos[2].text(0.3, -0.067, "barreira de perda", va="top", **kw)
eixos[2].text(9.7, 0.035, "prazo", ha="right", **kw)
salvar(fig, "cap09/barreiras")
