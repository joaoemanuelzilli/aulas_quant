"""Curvas de capital do mini backtest fora da amostra."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l07_backtest import bruto, liquido, atraso, R, rf as cdi

series = [("comprar e manter", R, CINZA, -5),
          ("CDI", cdi, CINZA, 0),
          ("modelo, sem custo", bruto, OCRE, 5),
          ("modelo, com custo", liquido, ROXO, 0),
          ("com custo e atraso", atraso, VERDE, 0)]
fig, ax = figura(altura=62)
for nome, r, cor, dy in series:
    curva = (1 + r).cumprod()
    ax.plot(curva.index, curva, color=cor,
            lw=1.3 if cor == ROXO else 0.95,
            ls=(0, (4, 2)) if nome == "CDI" else "-")
    ax.annotate(nome, (curva.index[-1], curva.iloc[-1]),
                xytext=(4, dy), textcoords="offset points",
                va="center", fontsize=7.2)
    print(nome, round(curva.iloc[-1], 3))
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 4])
ax.set_yticklabels(["0,5", "1", "2", "4"]); ax.minorticks_off()
ax.set_title("Valor de R$ 1 a partir de janeiro de 2019 "
             "(escala log)")
fim = R.index[-1]
ax.set_xlim(R.index[0], fim + pd.Timedelta(days=600))
ax.spines["bottom"].set_bounds(mpl.dates.date2num(R.index[0]),
                               mpl.dates.date2num(fim))
ax.set_xticks(pd.to_datetime([f"{a}-01-01"
                             for a in range(2019, 2027)]))
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
salvar(fig, "cap09/backtest")
