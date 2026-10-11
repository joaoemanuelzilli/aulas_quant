"""Preços em base 100: índices, dólar e CDI na mesma régua."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l11_eda import base100

bcb = pd.read_csv(DADOS / "c02_bcb.csv", index_col=0, parse_dates=True)
cdi = (1 + bcb["cdi"] / 100).cumprod().reindex(base100.index, method="ffill")
cdi = 100 * cdi / cdi.iloc[0]
series = {"S&P 500 (R$)": ("S&P 500 em reais", ROXO), "S&P 500 (US$)": ("S&P 500 em dólares", OCRE),
          "Dólar": ("dólar", VERDE), "Ibovespa": ("Ibovespa", AZUL)}
fig, ax = figura(altura=56)
ax.plot(cdi.index, cdi, color=CINZA, lw=1, ls=(0, (4, 2)))
fim = base100.index[-1]
for col, (nome, cor) in series.items():
    ax.plot(base100.index, base100[col], color=cor, lw=1.05)
    rotulo_final(ax, fim, base100[col].iloc[-1], nome, cor)
rotulo_final(ax, fim, cdi.iloc[-1], "CDI", CINZA)
ax.set_yscale("log")
ax.set_yticks([50, 100, 200, 500, 1000, 2000]); ax.set_yticklabels(["50", "100", "200", "500", "1.000", "2.000"])
ax.minorticks_off()
ax.set_title("Base 100 em 4 de janeiro de 2010 (escala log)")
ax.set_xlim(base100.index[0], fim + pd.Timedelta(days=1450))
ax.spines["bottom"].set_bounds(mpl.dates.date2num(base100.index[0]), mpl.dates.date2num(fim))
ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in range(2010, 2026, 3)])
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
salvar(fig, "cap02/base100")
bruto_cdi = (1 + bcb["cdi"] / 100).cumprod()
for ano in ("2016", "2020"):
    ini = base100.loc[ano].index[0]
    ib = base100["Ibovespa"]
    c = bruto_cdi.reindex(base100.index, method="ffill")
    print(f"início em {ini:%d/%m/%Y}: Ibovespa {100 * ib.iloc[-1] / ib[ini]:.0f}, CDI {100 * c.iloc[-1] / c[ini]:.0f}")
print("CDI base 100 final:", round(cdi.iloc[-1], 1), base100.iloc[-1].round(1).to_dict())
