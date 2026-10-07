"""Curva de capital do cruzamento 20/100 em PETR4 e drawdown."""
import sys; sys.path.insert(0, "..")
from estilo import *
from motor import carregar, cruzamento

preco, ret = carregar("PETR4.SA")
bt = cruzamento(preco, ret["cdi"]).loc["2011":"2025"]
ret = ret.loc["2011":"2025"]
series = {"comprar e segurar": (ret["ativo"], OCRE, 5),
          "CDI": (ret["cdi"], CINZA, 0),
          "cruzamento 20/100": (bt["retorno"], ROXO, -6)}
fig, (a1, a2) = figura(altura=82, nrows=2, sharex=True,
                       gridspec_kw={"height_ratios": [1.7, 1]})
fim = bt.index[-1]
for nome, (r, cor, dy) in series.items():
    curva = (1 + r).cumprod()
    a1.plot(curva.index, curva, color=cor, lw=1.1)
    a1.annotate(nome, (fim, curva.iloc[-1]), xytext=(4, dy),
                textcoords="offset points", va="center", fontsize=7.2)
    if nome != "CDI":
        dd = curva / curva.cummax().clip(lower=1) - 1
        a2.plot(dd.index, dd, color=cor, lw=0.9)
# faixa: dias em que a estratégia está comprada
a1.fill_between(bt.index, 0.17, 0.2, where=bt["posicao"] > 0,
                color=ROXO, lw=0, step="post")
a1.annotate("dias comprado", (fim, 0.184), xytext=(4, 0),
            textcoords="offset points", va="center", fontsize=7.2)
a1.set_yscale("log"); a1.set_ylim(0.17, 8)
a1.set_yticks([0.25, 0.5, 1, 2, 4])
a1.set_yticklabels(["0,25", "0,5", "1", "2", "4"]); a1.minorticks_off()
a1.set_title("Valor de R$ 1 aplicado em jan/2011 (escala log)")
a2.set_title("Drawdown")
pct(a2); a2.set_ylim(-0.9, 0.02)
a1.set_xlim(bt.index[0], fim + pd.Timedelta(days=1150))
for a in (a1, a2):
    a.spines["bottom"].set_bounds(mpl.dates.date2num(bt.index[0]),
                                  mpl.dates.date2num(fim))
a2.set_xticks([pd.Timestamp(f"{a}-01-01") for a in range(2012, 2026, 3)])
a2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
rotulos = {"comprar e segurar": (pd.Timestamp("2016-09-01"), -0.74),
           "cruzamento 20/100": (pd.Timestamp("2021-02-01"), -0.70)}
for nome, (x, y) in rotulos.items():
    a2.text(x, y, nome, fontsize=6.8)
salvar(fig, "cap07/curva_capital")
