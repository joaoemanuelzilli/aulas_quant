"""Figura do relatório-modelo: curva de capital e drawdown (underwater)."""
import sys; sys.path.insert(0, "..")
from estilo import *

d = pd.read_csv(DADOS / "c08_relatorio.csv", index_col=0, parse_dates=True)
curva = (1 + d).cumprod()
dd = curva / curva.cummax().clip(lower=1.0) - 1
fig, (a1, a2) = figura(altura=74, largura="larga", nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
for col, nome, cor in [("wf", "estratégia (walk-forward)", ROXO), ("petr4", "PETR4 comprada", OCRE), ("cdi", "CDI", CINZA)]:
    a1.plot(curva.index, curva[col], color=cor, lw=1.1)
    rotulo_final(a1, curva.index[-1], curva[col].iloc[-1], nome, cor)
a1.set_yscale("log"); a1.set_yticks([0.3, 1, 3, 10]); a1.set_yticklabels(["0,3", "1", "3", "10"]); a1.minorticks_off()
a1.set_title(f"Valor de R$ 1 aplicado em {curva.index[0]:%d/%m/%Y} (escala log)")
a2.fill_between(dd.index, dd["wf"], 0, color=ROXO, alpha=0.85)
a2.plot(dd.index, dd["petr4"], color=OCRE, lw=0.8)
a2.set_title("Drawdown: estratégia (área) e PETR4 comprada (linha)")
pct(a2); a2.set_ylim(-0.9, 0.02)
a2.set_xlim(curva.index[0], curva.index[-1] + pd.Timedelta(days=900))
a2.set_xticks([pd.Timestamp(f'{a}-01-01') for a in range(2014, 2027, 2)]); a2.set_xticklabels(range(2014, 2027, 2))
for a in (a1, a2):
    a.spines["bottom"].set_bounds(mpl.dates.date2num(curva.index[0]), mpl.dates.date2num(curva.index[-1]))
salvar(fig, "cap08/relatorio")
