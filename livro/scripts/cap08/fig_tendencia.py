"""Regra de tendência (média de 200 dias) em BOVA11: posição e curva de capital."""
import sys; sys.path.insert(0, "..")
from estilo import *
from motor import carregar, backtest
from regras import tendencia

p, ret, cdi = carregar("BOVA11.SA")
p, ret = p["BOVA11.SA"], ret["BOVA11.SA"]
alvo = tendencia(p, 200)
liq, _ = backtest(alvo, ret, cdi, 0.001)
bh, _ = backtest(pd.Series(1.0, index=p.index), ret, cdi, 0)

fig, (a1, a2) = figura(altura=80, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
pos = alvo.shift(1).fillna(0)
a1.fill_between(p.index, 0, 1, where=pos > 0, transform=a1.get_xaxis_transform(), color=LILAS, zorder=0)
a1.plot(p.index, p, color=ROXO, lw=0.9)
a1.plot(p.index, p.rolling(200).mean(), color=TINTA, lw=0.8)
a1.set_title("BOVA11 e média móvel de 200 dias (R$); em lilás, dias com posição comprada")
a1.annotate("média de 200 dias", (pd.Timestamp("2021-01-15"), p.rolling(200).mean()["2021-01-15"]), xytext=(pd.Timestamp("2020-06-01"), 150),
            fontsize=7, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
curvas = {"regra, líquida": ((1 + liq).cumprod(), ROXO), "BOVA11": ((1 + bh).cumprod(), OCRE), "CDI": ((1 + cdi).cumprod(), CINZA)}
for nome, (c, cor) in curvas.items():
    a2.plot(c.index, c, color=cor, lw=1.1)
    rotulo_final(a2, c.index[-1], c.iloc[-1], nome, cor)
a2.set_yscale("log"); a2.set_yticks([0.5, 1, 2, 4]); a2.set_yticklabels(["0,5", "1", "2", "4"]); a2.minorticks_off()
a2.set_title("Valor de R$ 1 (escala log)")
a2.set_xlim(p.index[0], p.index[-1] + pd.Timedelta(days=620))
for a in (a1, a2):
    a.spines["bottom"].set_bounds(mpl.dates.date2num(p.index[0]), mpl.dates.date2num(p.index[-1]))
salvar(fig, "cap08/tendencia")
print({k: round(v[0].iloc[-1], 2) for k, v in curvas.items()})
for a0, a1 in [("2010", "2015"), ("2016", "2025"), ("2020-01", "2020-03")]:   # números da legenda
    print(f"{a0} a {a1}: regra {(1 + liq[a0:a1]).prod() - 1:+.1%}   BOVA11 {(1 + bh[a0:a1]).prod() - 1:+.1%}")
