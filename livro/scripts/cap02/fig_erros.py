"""Quatro assinaturas: três problemas de dados e uma queda real."""
import sys; sys.path.insert(0, "..")
from estilo import *

p = pd.read_csv(DADOS / "bruto" / "precos_ajustados.csv", index_col=0, parse_dates=True)
bcb = pd.read_csv(DADOS / "c02_bcb.csv", index_col=0, parse_dates=True)
o = pd.read_csv(DADOS / "bruto" / "petr4_ohlcv.csv", index_col=0, parse_dates=True)
fig, eixos = figura(altura=82, nrows=2, ncols=2)


def par(ax, serie, ref, ini, fim, titulo, nome, nome_ref):
    d = pd.concat([serie, ref], axis=1, sort=True).loc[ini:fim].dropna()
    d = 100 * d / d.iloc[0]
    ax.plot(d.index, d.iloc[:, 1], color=CINZA, lw=1)
    ax.plot(d.index, d.iloc[:, 0], color=ROXO)
    ax.set_title(titulo)
    ax.xaxis.set_major_locator(mpl.dates.WeekdayLocator(byweekday=0, interval=2))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m"))
    return d


d = par(eixos[0, 0], p["ITUB4.SA"], p["^BVSP"], "2021-09-13", "2021-10-22", "Evento societário sem ajuste", "ITUB4", "Ibovespa")
eixos[0, 0].text(d.index[2], 81, "ITUB4, out/2021", fontsize=6.8)
eixos[0, 0].text(d.index[-1], 100.5, "Ibovespa", fontsize=6.8, color=CINZA, ha="right", va="bottom")
eixos[0, 0].set_ylim(78, 104)

y = p["BRL=X"].dropna().shift(-1)
d = par(eixos[0, 1], y, bcb["ptax_venda"], "2010-08-16", "2010-09-24", "Cotação errada que se desfaz", "BRL=X", "PTAX")
eixos[0, 1].text(d.index[1], 93.6, "dólar no Yahoo, 2010", fontsize=6.8)
eixos[0, 1].text(d.index[-1], 99.6, "PTAX", fontsize=6.8, color=CINZA, ha="right")
eixos[0, 1].set_ylim(93, 101.5)

d = par(eixos[1, 0], p["PETR4.SA"], p["^BVSP"], "2011-07-18", "2011-08-26", "Preço repetido pela fonte", "PETR4", "Ibovespa")
z = o.loc["2011-07-18":"2011-08-26"]; z = z.index[z.Volume == 0]
eixos[1, 0].axvspan(z[0], z[-1], color=LILAS, zorder=0)
eixos[1, 0].text(z[0] + (z[-1] - z[0]) / 2, 115.5, "PETR4 com volume zero", fontsize=6.8, ha="center")
eixos[1, 0].text(d.index[-1], 81, "PETR4, ago/2011", fontsize=6.8, ha="right")
eixos[1, 0].text(d.index[0], 90, "Ibovespa", fontsize=6.8, color=CINZA)
eixos[1, 0].set_ylim(78, 120)

d = par(eixos[1, 1], p["VALE3.SA"], p["^BVSP"], "2019-01-07", "2019-02-15", "Queda real", "VALE3", "Ibovespa")
eixos[1, 1].text(d.index[-1], 72, "VALE3, jan/2019", fontsize=6.8, ha="right")
eixos[1, 1].text(d.index[-1], 109.5, "Ibovespa", fontsize=6.8, color=CINZA, ha="right")
eixos[1, 1].set_ylim(68, 114)
salvar(fig, "cap02/erros")
