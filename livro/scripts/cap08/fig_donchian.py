"""Canal de Donchian em PETR4: teto de 20 dias, piso de 10 dias, entradas e saídas."""
import sys; sys.path.insert(0, "..")
from estilo import *
from motor import carregar_ohlc
from regras import donchian

o, ret, cdi = carregar_ohlc()
n = 20
teto = o["High"].rolling(n).max().shift(1)
piso = o["Low"].rolling(n // 2).min().shift(1)
alvo = donchian(o, n)
j = slice("2015-10-01", "2016-12-29")
c = o["Close"][j]
fig, ax = figura(altura=62)
ax.fill_between(c.index, 0, 1, where=alvo[j] > 0, transform=ax.get_xaxis_transform(), color=LILAS, zorder=0)
ax.step(c.index, teto[j], color=TINTA, lw=0.7, where="post")
ax.step(c.index, piso[j], color=CINZA, lw=0.7, where="post", ls=(0, (4, 2)))
ax.plot(c.index, c, color=ROXO, lw=1.1)
mud = alvo.diff()[j]
ent, sai = mud[mud > 0].index, mud[mud < 0].index
ax.plot(ent, c[ent], "^", color=VERDE, ms=5, zorder=5)
ax.plot(sai, c[sai], "v", color=OCRE, ms=5, zorder=5)
ax.set_title("PETR4: fechamento ajustado (R$), teto de 20 dias e piso de 10 dias")
ax.annotate("teto: máxima dos\n20 dias anteriores", (pd.Timestamp("2016-01-12"), teto["2016-01-12"]), xytext=(10, 26),
            textcoords="offset points", fontsize=6.8, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.annotate("piso: mínima dos\n10 dias anteriores", (pd.Timestamp("2016-09-01"), piso["2016-09-01"]), xytext=(18, -24),
            textcoords="offset points", fontsize=6.8, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.plot([], [], "^", color=VERDE, ms=5, label="compra"); ax.plot([], [], "v", color=OCRE, ms=5, label="venda")
ax.legend(loc="upper left", ncols=2, handletextpad=0.2, columnspacing=1.2)
ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m/%Y"))
virgula(ax, casas=1)
salvar(fig, "cap08/donchian")
print("entradas:", [str(d.date()) for d in ent]); print("saídas:", [str(d.date()) for d in sai])
