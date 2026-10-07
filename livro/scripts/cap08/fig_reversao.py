"""Reversão à média em BOVA11: bandas de Bollinger, z-score e operações em 2019-2020."""
import sys; sys.path.insert(0, "..")
from estilo import *
from motor import carregar, backtest, operacoes
from regras import zscore, reversao

p, ret, cdi = carregar("BOVA11.SA")
p, ret = p["BOVA11.SA"], ret["BOVA11.SA"]
x = np.log(p)
m, s = x.rolling(20).mean(), x.rolling(20).std()
z = zscore(p, 20)
alvo = reversao(p, 20, 2.0)
pos = alvo.shift(1).fillna(0)
j = slice("2019-09-01", "2020-09-30")

fig, (a1, a2) = figura(altura=80, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
for a in (a1, a2):
    a.fill_between(p[j].index, 0, 1, where=pos[j] > 0, transform=a.get_xaxis_transform(), color=LILAS, zorder=0)
a1.fill_between(p[j].index, np.exp(m - 2 * s)[j], np.exp(m + 2 * s)[j], color=CINZA_CLARO, alpha=0.55, zorder=1)
a1.plot(p[j].index, np.exp(m)[j], color=CINZA, lw=0.8)
a1.plot(p[j].index, p[j], color=ROXO, lw=1.1)
a1.set_title("BOVA11 (R$), média de 20 dias e bandas de 2 desvios; em lilás, dias comprado")
a2.plot(z[j].index, z[j], color=ROXO, lw=1.0)
fim = p[j].index[-1]
# as linhas de referência param no último dado, para não cortar os rótulos
a2.hlines(0, p[j].index[0], fim, color=TINTA, lw=0.6)
a2.hlines(-2, p[j].index[0], fim, color=OCRE, lw=0.9, ls=(0, (4, 3)))
a2.annotate("entrada:\nz < −2", (fim, -2), xytext=(5, 0), textcoords="offset points", fontsize=6.8, va="center")
a2.annotate("saída:\nz ≥ 0", (fim, 0), xytext=(5, 0), textcoords="offset points", fontsize=6.8, va="center")
a2.set_xlim(p[j].index[0] - pd.Timedelta(days=6), fim + pd.Timedelta(days=48))
for a in (a1, a2):
    a.spines["bottom"].set_bounds(mpl.dates.date2num(p[j].index[0]), mpl.dates.date2num(fim))
a2.set_title("z-score do log-preço")
a2.set_ylim(-4.6, 3.2)
a2.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2)); a2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m/%Y"))
salvar(fig, "cap08/reversao")
mud = alvo.diff()
print("operações com sinal de compra em fev-mar/2020 (datas do sinal, no fechamento):")
for d in mud[(mud > 0)]["2020-02":"2020-03"].index:
    fim = mud[d:][mud[d:] < 0].index[0]
    r = (1 + ret[d:fim].iloc[1:]).prod() - 1
    print(f"  compra {d.date()}  venda {fim.date()}  retorno bruto {r:+.1%}  z mínimo {z[d:fim].min():.2f}")
liq = __import__("motor").backtest(alvo, ret, cdi)[0]
mensal = (1 + liq).resample("ME").prod() - 1
print(f"assimetria mensal: {mensal.skew():+.2f}; sem março de 2020: {mensal.drop(mensal['2020-03'].index).skew():+.2f}")
