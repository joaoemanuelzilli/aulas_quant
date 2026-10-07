"""VaR histórico com janelas de 250 e 1000 dias: o efeito fantasma."""
import sys; sys.path.insert(0, "..")
from estilo import *
from carteira import perda

L = perda / 1000
v250 = L.rolling(250).quantile(0.99).shift(1)
v1000 = L.rolling(1000).quantile(0.99).shift(1)
ini, fim = "2019-07", "2022-06"
fig, (a1, a2) = figura(altura=62, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
s = L[ini:fim]
a1.bar(s.index, s.clip(lower=0), width=1.6, color=CINZA)
a1.set_title("Perdas diárias da carteira (R$ mil)")
a2.plot(v1000[ini:fim], color=OCRE)
a2.plot(v250[ini:fim], color=ROXO)
a2.set_title("VaR histórico de 99% (R$ mil)")
x = v250[ini:fim]
a2.annotate("250 dias", (x.index[-1], x.iloc[-1]), xytext=(4, -3), textcoords="offset points", fontsize=7.2, va="center")
a2.annotate("1000 dias", (x.index[-1], v1000[ini:fim].iloc[-1]), xytext=(4, 3), textcoords="offset points", fontsize=7.2, va="center")
a2.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=(1, 7)))
a2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m/%Y"))
salto = x.diff()
sobe, desce = salto.idxmax(), salto.idxmin()
a2.annotate(f"sobe em\n{sobe:%d/%m/%Y}", (sobe, x[sobe]), xytext=(-12, -4), textcoords="offset points", fontsize=6.8, ha="right", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a2.annotate(f"cai em {desce:%d/%m/%Y},\nsem nada ter acontecido\nno mercado", (desce, x[desce] + 25), xytext=(14, 8), textcoords="offset points",
            fontsize=6.8, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a2.set_xlim(s.index[0], s.index[-1] + pd.Timedelta(days=130))
for a in (a1, a2):
    a.spines["bottom"].set_bounds(mpl.dates.date2num(s.index[0]), mpl.dates.date2num(s.index[-1]))
a2.set_ylim(0, None)
salvar(fig, "cap05/fantasma")
print(sobe, x[:sobe].iloc[-2:], desce, x[:desce].iloc[-2:], salto.nsmallest(4), salto.nlargest(4))
print("v1000 antes/depois", v1000["2020-02-28"], v1000["2020-04-30"], "v250", v250["2020-02-28"], v250["2020-04-30"], v250["2021-04-30"])
