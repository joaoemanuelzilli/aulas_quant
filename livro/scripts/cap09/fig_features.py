"""Três famílias de features em um ano de PETR4."""
import sys; sys.path.insert(0, "..")
from estilo import *
from features import carregar, rsi, mme

d = carregar()
p = d["Close"]
mm, dp = p.rolling(20).mean(), p.rolling(20).std()
sup, inf = mm + 2 * dp, mm - 2 * dp
ifr = rsi(p, 14)
macd = (mme(p, 12) - mme(p, 26)) / p
hist = macd - mme(macd, 9)
j = slice("2024-01-01", "2024-12-31")

fig, (a1, a2, a3) = figura(
    altura=84, nrows=3, sharex=True,
    gridspec_kw={"height_ratios": [1.5, 1, 1]})
a1.fill_between(p[j].index, inf[j], sup[j], color=LILAS)
a1.plot(mm[j].index, mm[j], color=CINZA, lw=0.8)
a1.plot(p[j].index, p[j], color=ROXO, lw=1.1)
a1.set_title("Preço ajustado (R$), média de 20 dias e bandas de "
             "Bollinger")
a2.plot(ifr[j].index, ifr[j], color=ROXO, lw=1.0)
for nivel in (30, 70):
    a2.axhline(nivel, color=CINZA, lw=0.6, ls=(0, (4, 3)))
a2.set_ylim(10, 90); a2.set_yticks([30, 50, 70])
a2.set_title("RSI de 14 dias")
h = hist[j]
a3.bar(h.index, h, width=1.0, color=np.where(h >= 0, ROXO, OCRE))
a3.axhline(0, color=TINTA, lw=0.5)
a3.set_title("Histograma do MACD, em fração do preço")
pct(a3, casas=1)
a3.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
a3.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m/%Y"))
salvar(fig, "cap09/features")
fora = ((p[j] > sup[j]) | (p[j] < inf[j])).mean()
i = ifr[j]
print(f"2024: RSI mín {i.min():.1f} em {i.idxmin():%d/%m}"
      f", máx {i.max():.1f} em {i.idxmax():%d/%m};"
      f" dias fora das bandas {fora:.1%};"
      f" RSI<30: {(i < 30).sum()}, >70: {(i > 70).sum()}")
