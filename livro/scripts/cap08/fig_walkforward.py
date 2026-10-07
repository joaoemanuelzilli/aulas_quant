"""Walk-forward da regra de Donchian em PETR4: n escolhido por janela e curvas fora da amostra."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l07_walkforward import walk_forward, rets, ret, cdi, sharpe, TREINO

wf, tab, _ = walk_forward()
wfa, taba, _ = walk_forward(ancorada=True)
n_fixo = rets.iloc[TREINO:].apply(sharpe).idxmax()
fixo = rets[n_fixo].loc[wf.index]
bh = ret.loc[wf.index]

fig, (a1, a2) = figura(altura=84, nrows=2, gridspec_kw={"height_ratios": [0.8, 1.25]})
anos = tab["ano"].tolist() + [2026]
a1.step(anos, tab["n"].tolist() + [tab["n"].iloc[-1]], where="post", color=ROXO, lw=1.3)
a1.step(anos, taba["n"].tolist() + [taba["n"].iloc[-1]], where="post", color=OCRE, lw=1.1, ls=(0, (4, 2)))
a1.set_title("Valor de n escolhido para cada ano de teste")
a1.set_xticks(range(2013, 2027, 2)); a1.set_xlim(2013, 2030.3); a1.set_ylim(0, 110); a1.set_yticks([20, 40, 60, 80, 100])
a1.annotate("rolante", (2026, tab["n"].iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.2, va="center")
a1.annotate("ancorada", (2026, taba["n"].iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.2, va="center")
a1.spines["bottom"].set_bounds(2013, 2026)
curvas = [(f"n = {n_fixo} fixo, escolhido depois", fixo, CINZA, 5), ("walk-forward rolante", wf, ROXO, -5), ("PETR4 comprada", bh, OCRE, 0)]
for nome, r, cor, dy in curvas:
    c = (1 + r).cumprod()
    a2.plot(c.index, c, color=cor, lw=1.1)
    a2.annotate(nome, (c.index[-1], c.iloc[-1]), xytext=(4, dy), textcoords="offset points", fontsize=7.2, va="center")
a2.set_yscale("log"); a2.set_yticks([0.3, 1, 3, 10, 30]); a2.set_yticklabels(["0,3", "1", "3", "10", "30"]); a2.minorticks_off()
a2.set_title("Valor de R$ 1 no período fora da amostra (escala log)")
a2.set_xlim(wf.index[0], wf.index[-1] + pd.Timedelta(days=1750))
a2.set_xticks([pd.Timestamp(f'{a}-01-01') for a in range(2014, 2027, 2)]); a2.set_xticklabels(range(2014, 2027, 2))
a2.spines["bottom"].set_bounds(mpl.dates.date2num(wf.index[0]), mpl.dates.date2num(wf.index[-1]))
salvar(fig, "cap08/walkforward")
print({n: round((1 + r).prod(), 2) for n, r, _, _ in curvas})
