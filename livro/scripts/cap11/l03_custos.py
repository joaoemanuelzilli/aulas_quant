import numpy as np
import pandas as pd

ATIVOS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA",
          "WEGE3.SA", "ABEV3.SA", "BBAS3.SA"]
p = pd.read_csv("../../dados/precos_ajustados.csv", index_col=0,
                parse_dates=True)[ATIVOS].dropna()
R = p.pct_change()

# momentum 12-1: retorno de 12 meses, pulando o mês mais recente
sinal = (p.shift(21) / p.shift(252) - 1).dropna()
d = sinal.sub(sinal.mean(axis=1), axis=0)
alvo = d.div(d.abs().sum(axis=1), axis=0)   # soma 0, soma |w| = 1


def simular(alvo, R, rebalancear):
    """Retorno bruto e giro por dia, com capital constante 1."""
    v = np.zeros(alvo.shape[1])             # posições em R$
    bruto, giro = [], []
    for dia, w in alvo.iterrows():
        r = R.loc[dia].values
        bruto.append(v @ r)                 # resultado do dia
        v = v * (1 + r)                     # posições derivam
        negocia = rebalancear(dia)
        giro.append(np.abs(w.values - v).sum() if negocia else 0)
        v = w.values if negocia else v      # no fechamento
    return (pd.Series(bruto, alvo.index),
            pd.Series(giro, alvo.index))


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)


meses = alvo.index.to_period("M")
fim_de_mes = set(alvo.index[meses != np.roll(meses, -1)])
versoes = {"diário": lambda dia: True,
           "mensal": lambda dia: dia in fim_de_mes}
resultado = {n: simular(alvo, R, f) for n, f in versoes.items()}

if __name__ == "__main__":
    print(f"{alvo.index[0]:%Y-%m-%d} a {alvo.index[-1]:%Y-%m-%d}")
    print("rebalanceamento         diário            mensal")
    giros = [g.mean() * 252 for _, g in resultado.values()]
    print("giro anual (x capital)  {:6.1f}            {:6.1f}"
          .format(*giros))
    print("custo por ponta    retorno  Sharpe   retorno  Sharpe")
    for bps in (0, 5, 10, 20):
        linha = f"{bps:9d} pb      "
        for bruto, giro in resultado.values():
            liq = bruto - bps / 1e4 * giro
            ret = liq.mean() * 252
            linha += f"{ret:8.2%} {sharpe(liq):7.2f}  "
        print(linha.rstrip())
    eqs = [b.sum() / g.sum() * 1e4 for b, g in resultado.values()]
    print("custo de equilíbrio  {:6.1f} pb         {:6.1f} pb"
          .format(*eqs))
