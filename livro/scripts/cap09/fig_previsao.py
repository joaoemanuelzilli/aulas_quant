"""Vantagem acumulada do futuro sobre o passeio aleatório."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l07_previsao import previsoes

fig, ax = figura(altura=56)
for h, cor in [(1, ROXO), (2, OCRE), (3, VERDE)]:
    p = previsoes(h)
    ganho = ((p.passeio - p.real) ** 2 - (p.futuro - p.real) ** 2).cumsum()
    t = ganho.index.to_timestamp(how="end")
    ax.plot(t, ganho, color=cor)
    rotulo_final(ax, t[-1], ganho.iloc[-1], f"{h} {'mês' if h == 1 else 'meses'}", cor)
ax.axhline(0, color=CINZA, lw=0.5)
ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), color=CINZA_CLARO, alpha=0.6, zorder=0)
ax.set_title("Soma dos erros quadráticos do passeio menos os do futuro")
ax.set_xlim(pd.Timestamp("2014-10-01"), pd.Timestamp("2025-09-01"))
virgula(ax, casas=2)
salvar(fig, "cap09/previsao")
