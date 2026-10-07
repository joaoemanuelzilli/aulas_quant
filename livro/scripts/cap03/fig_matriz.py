"""Matriz de correlação de retornos diários, 2015 a 2024."""
import sys; sys.path.insert(0, "..")
from estilo import *

nomes = {"PETR4.SA": "PETR4", "VALE3.SA": "VALE3", "ITUB4.SA": "ITUB4", "BBAS3.SA": "BBAS3",
         "ABEV3.SA": "ABEV3", "WEGE3.SA": "WEGE3", "MGLU3.SA": "MGLU3", "^GSPC": "S&P 500"}
p = precos(*nomes, inicio="2015", fim="2024").dropna()      # só dias comuns a B3 e NYSE
r = p.pct_change().dropna().rename(columns=nomes)
c = r.corr()
k = len(c)
fig, ax = figura(altura=78)
ax.imshow(c, cmap=DIV, vmin=-1, vmax=1)
for i in range(k):
    for j in range(k):
        v = c.iloc[i, j]
        ax.text(j, i, f"{v:.2f}".replace(".", ",").replace("-", "−"), ha="center", va="center",
                fontsize=6.6, color="white" if abs(v) > 0.6 else TINTA)
ax.set_xticks(range(k)); ax.set_xticklabels(c.columns, rotation=40, ha="left")
ax.set_yticks(range(k)); ax.set_yticklabels(c.index)
ax.xaxis.tick_top()
ax.tick_params(length=0, labelcolor=TINTA)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
salvar(fig, "cap03/matriz_corr")
print(len(r), "dias comuns")
print(c.round(2).to_string())
acoes = c.iloc[:7, :7].values
fora = acoes[np.triu_indices(7, 1)]
print("entre as 7 ações: média", fora.mean().round(3), "mín", fora.min().round(3), "máx", fora.max().round(3))
