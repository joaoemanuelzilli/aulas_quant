"""Backtest do VaR histórico de 99%: violações ao longo do tempo."""
import sys; sys.path.insert(0, "..")
from estilo import *
from carteira import perda

L = perda / 1000
var = L.rolling(250).quantile(0.99).shift(1)
L, var = L[var.notna()]["2011":"2025"], var.dropna()["2011":"2025"]
viol = L > var
fig, (a1, a2) = figura(altura=70, largura="larga", nrows=2, gridspec_kw={"height_ratios": [1.7, 1]})
a1.bar(L.index, L.clip(lower=0), width=2.5, color=CINZA_CLARO)
a1.plot(var, color=ROXO, lw=1)
a1.plot(L[viol].index, L[viol], "o", color=OCRE, ms=2.8, zorder=3)
a1.set_title("Perdas diárias, VaR histórico de 99% (janela de 250 dias) e violações (R$ mil)")
a1.set_xlim(L.index[0], L.index[-1])
a1.text(pd.Timestamp("2012-03-01"), 118, "linha: VaR previsto na véspera", fontsize=7)
a1.text(pd.Timestamp("2012-03-01"), 104, "pontos: dias em que a perda superou o VaR", fontsize=7)
por_ano = viol.groupby(viol.index.year).sum()
dias = viol.groupby(viol.index.year).size()
a2.bar(por_ano.index, por_ano.values, width=0.62, color=ROXO)
a2.plot(dias.index, dias.values * 0.01, color=TINTA, lw=0, marker="_", ms=13, mew=1)
a2.set_title("Violações por ano (traço: número esperado, 1% dos pregões)")
a2.set_xticks(por_ano.index); a2.set_xlim(2010.4, 2025.6)
for x, y in por_ano.items():
    a2.text(x, max(y, 2.5) + 0.3, str(y), ha="center", fontsize=6.8)
a2.set_ylim(0, por_ano.max() + 2)
salvar(fig, "cap05/backtest")
print(por_ano.to_dict(), viol.sum(), len(viol)); print(L[viol]["2020"])
