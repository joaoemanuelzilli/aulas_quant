"""Volatilidade móvel e distância ao pico: o risco muda no tempo."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l11_eda import vol, queda

series = {"Ibovespa": ROXO, "S&P 500 (US$)": OCRE, "Dólar": VERDE}
fig, (a1, a2) = figura(altura=70, nrows=2, sharex=True)
for col, cor in series.items():
    a1.plot(vol.index, vol[col], color=cor, lw=0.9)
a1.set_title("Volatilidade anualizada em janela móvel de 63 pregões")
pct(a1)
kw = dict(fontsize=6.8, va="center")
a1.text(pd.Timestamp("2020-09-15"), 0.76, "Ibovespa", **kw)
a1.text(pd.Timestamp("2020-09-15"), 0.60, "S&P 500 (US$)", **kw)
a1.text(pd.Timestamp("2020-10-20"), 0.33, "dólar", **kw)
for col, cor in list(series.items())[:2]:
    a2.plot(queda.index, queda[col], color=cor, lw=0.9)
a2.axhline(0, color=TINTA, lw=0.6)
a2.set_title("Drawdown: distância ao pico anterior")
pct(a2)
i = queda["Ibovespa"].idxmin(); s = queda["S&P 500 (US$)"].idxmin()
a2.annotate(f"Ibovespa\n{queda['Ibovespa'].min():.1%}".replace(".", ","), (i, queda["Ibovespa"].min()),
            xytext=(-8, 2), textcoords="offset points", fontsize=6.8, ha="right", va="center")
a2.annotate(f"S&P 500 (US$)\n{queda['S&P 500 (US$)'].min():.1%}".replace(".", ","), (s, queda["S&P 500 (US$)"].min()),
            xytext=(34, -12), textcoords="offset points", fontsize=6.8, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA, shrinkB=2))
salvar(fig, "cap02/risco")
