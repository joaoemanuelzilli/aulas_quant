"""Mapa dos valores ausentes no conjunto congelado."""
import sys; sys.path.insert(0, "..")
from estilo import *

nomes = {"PETR4.SA": "PETR4", "MGLU3.SA": "MGLU3", "^BVSP": "Ibovespa", "^GSPC": "S&P 500", "BRL=X": "Dólar (BRL=X)"}
p = pd.read_csv(DADOS / "bruto" / "precos_ajustados.csv", index_col=0, parse_dates=True)[list(nomes)]
fig, (a1, a2) = figura(altura=66, nrows=2, gridspec_kw={"height_ratios": [1, 1]})
for ax, janela in ((a1, slice(None)), (a2, slice("2024-01-01", "2024-12-31"))):
    q = p.loc[janela]
    for i, t in enumerate(nomes):
        f = q.index[q[t].isna()]
        ax.vlines(f, i - 0.33, i + 0.33, color=ROXO, lw=0.55 if ax is a1 else 1.3)
        ax.hlines(i, q.index[0], q.index[-1], color=CINZA_CLARO, lw=0.5, zorder=0)
    ax.set_yticks(range(len(nomes))); ax.set_yticklabels(nomes.values())
    ax.set_ylim(len(nomes) - 0.4, -0.6)
    ax.grid(False); ax.tick_params(axis="y", colors=TINTA)
    ax.spines["left"].set_visible(False)
a1.set_title("Arquivo bruto: cada traço é uma data sem cotação, 2010 a 2025")
a2.set_title("O mesmo mapa, só 2024")
a2.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
a2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m/%Y"))
kw = dict(fontsize=6.6, ha="center", va="bottom", color=TINTA)
a2.text(pd.Timestamp("2024-02-12"), -0.42, "Carnaval", **kw)
a2.text(pd.Timestamp("2024-11-18"), -0.42, "feriados de\nnovembro", **kw)
a2.text(pd.Timestamp("2024-11-28"), 3.48, "Ação de Graças", fontsize=6.6, ha="center", va="top")
a2.text(pd.Timestamp("2024-07-04"), 3.48, "4 de julho", fontsize=6.6, ha="center", va="top")
salvar(fig, "cap02/faltantes")
q = p.loc["2024"]
for t in nomes:
    print(t, [f"{d:%d/%m}" for d in q.index[q[t].isna()]])
