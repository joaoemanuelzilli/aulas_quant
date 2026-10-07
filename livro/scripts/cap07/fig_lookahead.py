"""O mesmo sinal com e sem defasagem: duas regras em PETR4."""
import sys; sys.path.insert(0, "..")
from estilo import *
from motor import carregar, cruzamento

preco, ret = carregar("PETR4.SA")
r, rf = ret["ativo"], ret["cdi"]


def regra_rapida(defasagem):
    sinal = (preco > preco.rolling(5).mean()).astype(float)
    pos = sinal.shift(defasagem).fillna(0.0)
    giro = pos.diff().abs().fillna(0.0)
    return pos * r + (1 - pos) * rf - 0.001 * giro


casos = [("Fechamento acima da média de 5 dias", regra_rapida),
         ("Cruzamento de médias 20/100",
          lambda a: cruzamento(preco, rf, defasagem=a)["retorno"])]
fig, eixos = figura(altura=56, ncols=2)
for ax, (titulo, f) in zip(eixos, casos):
    for defasagem, cor, nome in [(0, OCRE, "sem shift"), (1, ROXO, "shift(1)")]:
        curva = (1 + f(defasagem).loc["2011":"2025"]).cumprod()
        ax.plot(curva.index, curva, color=cor, lw=1.1)
        rotulo_final(ax, curva.index[-1], curva.iloc[-1], nome, cor)
        print(titulo, defasagem, curva.iloc[-1])
    ax.set_yscale("log"); ax.set_title(titulo, fontsize=7.6)
    ax.set_xlim(curva.index[0], curva.index[-1] + pd.Timedelta(days=2300))
    ax.spines["bottom"].set_bounds(mpl.dates.date2num(curva.index[0]),
                                   mpl.dates.date2num(curva.index[-1]))
    ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in (2012, 2016, 2020, 2024)])
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.minorticks_off()
eixos[0].set_yticks([1, 1e3, 1e6, 1e9])
eixos[0].set_yticklabels(["1", "mil", "1 milhão", "1 bilhão"])
eixos[1].set_yticks([0.5, 1, 2, 4]); eixos[1].set_yticklabels(["0,5", "1", "2", "4"])
eixos[0].set_ylabel("valor de R$ 1 (escala log)")
salvar(fig, "cap07/lookahead")
