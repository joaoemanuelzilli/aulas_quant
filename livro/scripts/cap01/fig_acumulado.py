"""Crescimento de R$ 1 em escala logarítmica."""
import sys; sys.path.insert(0, "..")
from estilo import *

nomes = {"WEGE3.SA": "WEGE3", "PETR4.SA": "PETR4", "VALE3.SA": "VALE3",
         "BOVA11.SA": "BOVA11", "ABEV3.SA": "ABEV3"}
p = precos(*nomes, inicio="2015", fim="2024").dropna()
ac = p / p.iloc[0]
fig, ax = figura(altura=66)
for (t, n), c in zip(nomes.items(), CATEGORICAS):
    ax.plot(ac.index, ac[t], color=c, lw=1.1)
    rotulo_final(ax, ac.index[-1], ac[t].iloc[-1], n, c)
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0,5", "1", "2", "5", "10"]); ax.minorticks_off()
ax.set_title("Valor de R$ 1 investido em jan/2015 (escala log)")
ax.set_xlim(ac.index[0], ac.index[-1] + pd.Timedelta(days=330))
ax.spines["bottom"].set_bounds(mpl.dates.date2num(ac.index[0]), mpl.dates.date2num(ac.index[-1]))
salvar(fig, "cap01/acumulado")
