"""Curvas de patrimônio fora da amostra, com e sem MGLU3."""
import sys; sys.path.insert(0, "..")
from estilo import *
from experimento import carregar_cdi

fig, eixos = figura(altura=64, largura="larga", ncols=2, sharey=True)
arqs = ["c06_fora_da_amostra.csv", "c06_fora_da_amostra_sem_mglu.csv"]
titulos = ["Sete ações", "Seis ações (sem MGLU3)"]
regras = ["máx. Sharpe", "1/N", "mín. variância"]
for ax, arq, titulo in zip(eixos, arqs, titulos):
    d = pd.read_csv(DADOS / arq, index_col=0, parse_dates=True)
    curva = (1 + d[regras]).cumprod()
    cdi = (1 + carregar_cdi(d.index)).cumprod()
    ax.plot(cdi.index, cdi, color=CINZA, lw=0.9)
    fins = {"CDI": cdi.iloc[-1]}
    for regra, cor in zip(regras, CATEGORICAS):
        ax.plot(curva.index, curva[regra], color=cor, lw=1.0)
        fins[regra] = curva[regra].iloc[-1]
    # rótulos no fim das linhas, afastados quando se sobrepõem
    ys = dict(sorted(fins.items(), key=lambda kv: kv[1]))
    ant = 0
    for nome, v in ys.items():
        pos = max(v, ant * 1.22)
        ax.annotate(nome, (curva.index[-1], pos), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7)
        ant = pos
    ax.set_yscale("log"); ax.set_title(titulo)
    ax.set_xlim(curva.index[0], curva.index[-1] + pd.Timedelta(days=1250))
    ax.spines["bottom"].set_bounds(mpl.dates.date2num(curva.index[0]),
                                   mpl.dates.date2num(curva.index[-1]))
    ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in range(2016, 2027, 2)],
                  [str(a) for a in range(2016, 2027, 2)])
    print(titulo, {k: round(float(v), 2) for k, v in fins.items()})
eixos[0].set_yticks([1, 2, 5, 10, 20, 40])
eixos[0].set_yticklabels(["1", "2", "5", "10", "20", "40"])
eixos[0].minorticks_off(); eixos[1].minorticks_off()
eixos[0].set_ylabel("valor de R$ 1 (escala log)")
salvar(fig, "cap06/fora_da_amostra")
