"""Crescimento de R$ 1 em escala logarítmica."""
import sys; sys.path.insert(0, "..")
from estilo import *

nomes = {"WEGE3.SA": "WEGE3", "PETR4.SA": "PETR4", "VALE3.SA": "VALE3",
         "BOVA11.SA": "BOVA11", "ABEV3.SA": "ABEV3"}
# mesma amostra do Código l04_anualizar.py: base no último fechamento de 2014
p = precos(*nomes, inicio="2014", fim="2024").dropna()
p = p.loc[p.loc["2014"].index[-1]:]
ac = p / p.iloc[0]
fig, ax = figura(altura=66)
for (t, n), c in zip(nomes.items(), CATEGORICAS):
    ax.plot(ac.index, ac[t], color=c, lw=1.1)
    # PETR4 e WEGE3 terminam quase juntas: afasta os dois rótulos
    dy = {"PETR4": 4, "WEGE3": -4}.get(n, 0)
    ax.annotate(n, (ac.index[-1], ac[t].iloc[-1]), xytext=(4, dy),
                textcoords="offset points", color=TINTA, va="center", fontsize=7.2)
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0,5", "1", "2", "5", "10"]); ax.minorticks_off()
ax.set_title("Valor de R$ 1 aplicado no fim de 2014 (escala log)")
ax.set_xlim(ac.index[0], ac.index[-1] + pd.Timedelta(days=330))
ax.spines["bottom"].set_bounds(mpl.dates.date2num(ac.index[0]), mpl.dates.date2num(ac.index[-1]))
salvar(fig, "cap01/acumulado")
print(f"base {ac.index[0]:%d/%m/%Y}; fim {ac.index[-1]:%d/%m/%Y}:", ac.iloc[-1].round(2).to_dict())
