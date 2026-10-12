"""Dois episódios: o Yahoo contra o fechamento oficial da B3."""
import sys; sys.path.insert(0, "..")
from estilo import *
from conferir_fonte import comparar, ler_fontes

b3, yahoo, volume, eventos = ler_fontes()
fig, (a1, a2) = figura(altura=50, ncols=2)


def painel(ax, papel, ini, fim, titulo):
    cortes = list(eventos.loc[eventos["papel"] == papel, "data_ex"])
    t = comparar(b3[papel], yahoo[papel + ".SA"], volume[papel + ".SA"], cortes).loc[ini:fim]
    ax.plot(t.index, t["oficial"], color=ROXO, marker="o", ms=2.6, lw=1.1)
    ax.plot(t.index, t["yahoo"], color=OCRE, marker="s", ms=2.6, lw=1.1, ls=(0, (3, 2)))
    ax.set_title(titulo)
    ax.xaxis.set_major_locator(mpl.dates.WeekdayLocator(byweekday=0, interval=1))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m"))
    virgula(ax, casas=1)
    return t


t = painel(a1, "PETR4", "2011-07-25", "2011-08-19", "PETR4, agosto de 2011 (R$)")
a1.text(pd.Timestamp("2011-08-06"), 23.75, "Yahoo: preço repetido,\nvolume zero", fontsize=6.8, ha="center", va="bottom")
a1.text(pd.Timestamp("2011-07-30"), 19.6, "B3: fechamento\noficial", fontsize=6.8, ha="center", va="center")
a1.set_ylim(18.2, 25.2)
t = painel(a2, "ITUB4", "2013-05-13", "2013-05-24", "ITUB4, maio de 2013 (R$)")
a2.annotate("Yahoo em 20/05:\n9,1% abaixo\ndo oficial", (pd.Timestamp("2013-05-20"), t.loc["2013-05-20", "yahoo"]),
            xytext=(12, 8), textcoords="offset points", fontsize=6.8, va="center", ha="left")
salvar(fig, "cap02/episodios")
print(t.loc["2013-05-20", ["oficial", "yahoo", "nivel"]].round(3).to_dict())
