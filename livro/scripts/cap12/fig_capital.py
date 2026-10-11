"""Curva de capital e drawdown, dentro e fora da amostra."""
import sys; sys.path.insert(0, "..")
from estilo import *
from projeto import *

px, ret, cdi = carregar()
cart = [("momentum", alvo_momentum(px, 12, 2), ROXO, 1.3),
        ("pesos iguais", alvo_fixo(px, ACOES), OCRE, 1.0),
        ("BOVA11", alvo_fixo(px, ["BOVA11.SA"]), CINZA, 1.0)]
fig, eixos = figura(altura=78, largura="larga", nrows=2, ncols=2,
                    sharey="row", gridspec_kw={
                        "width_ratios": [7, 6],
                        "height_ratios": [1.9, 1]})
titulos = ["Dentro da amostra (treino, 2013 a 2019)",
           "Fora da amostra (teste, 2020 a 2025)"]
for col, per in enumerate([TREINO, TESTE]):
    a1, a2 = eixos[0, col], eixos[1, col]
    fins = []
    for nome, alvo, cor, lw in cart:
        r, _, _ = simular(alvo, ret, cdi, per)
        cota = (1 + r).cumprod()
        a1.plot(cota.index, cota, color=cor, lw=lw)
        a2.plot(r.index, drawdown(r), color=cor, lw=lw * 0.8)
        fins.append((cota.iloc[-1], nome))
        print(per[0][:4], nome, round(cota.iloc[-1], 2),
              round(drawdown(r).min(), 3))
    # afasta rótulos que cairiam um sobre o outro (escala log)
    fins.sort()
    ys = [f[0] for f in fins]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] * 1.32)
    for y, (_, nome) in zip(ys, fins):
        rotulo_final(a1, cota.index[-1], y, nome, TINTA)
    a1.set_yscale("log")
    a1.set_yticks([0.5, 1, 2, 5, 10])
    a1.set_yticklabels(["0,5", "1", "2", "5", "10"])
    a1.minorticks_off()
    a1.set_title(titulos[col])
    a1.axhline(1, color=CINZA, lw=0.5)
    a1.tick_params(labelbottom=False)
    pct(a2)
    a2.set_ylim(-0.62, 0.02)
    a2.set_yticks([-0.6, -0.4, -0.2, 0])
    fim = cota.index[-1] + pd.Timedelta(days=520)
    for a in (a1, a2):
        a.set_xlim(cota.index[0], fim)
        anos = range(cota.index[0].year + 1, cota.index[-1].year + 1)
        a.set_xticks([pd.Timestamp(f"{y}-01-01") for y in anos],
                     [str(y) for y in anos])
        a.spines["bottom"].set_bounds(
            mpl.dates.date2num(cota.index[0]),
            mpl.dates.date2num(cota.index[-1]))
eixos[0, 0].set_ylabel("valor de R$ 1 (escala log)")
eixos[1, 0].set_ylabel("drawdown")
salvar(fig, "cap12/capital")
