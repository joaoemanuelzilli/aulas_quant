import numpy as np
import pandas as pd
from motor import carregar_ohlc, backtest, tabela
from regras import donchian

ohlc, ret, cdi = carregar_ohlc()

GRADE = range(10, 105, 5)       # candidatos a n: 10, 15, ..., 100
TREINO, TESTE = 756, 252        # três anos e um ano, em pregões


def excesso(r):                 # retorno anual acima do CDI
    return (r - cdi.loc[r.index]).mean() * 252


def sharpe(r):
    ex = r - cdi.loc[r.index]
    return ex.mean() / ex.std() * np.sqrt(252)


# posição e retorno líquido de cada candidato, amostra inteira
alvos = pd.DataFrame({n: donchian(ohlc, n) for n in GRADE})
rets = alvos.apply(lambda a: backtest(a, ret, cdi)[0])


def walk_forward(ancorada=False):
    costura = pd.Series(0.0, index=ohlc.index)
    linhas = []
    for ini in range(TREINO, len(ohlc), TESTE):
        comeco = 0 if ancorada else ini - TREINO
        treino = rets.iloc[comeco:ini]
        n = treino.apply(sharpe).idxmax()   # só dados passados
        fim = min(ini + TESTE, len(ohlc))
        costura.iloc[ini:fim] = alvos[n].iloc[ini:fim]
        teste = rets[n].iloc[ini:fim]
        linhas.append([ohlc.index[ini].year, n, sharpe(treino[n]),
                       sharpe(teste), excesso(treino[n]),
                       excesso(teste)])
    r, _ = backtest(costura, ret, cdi)      # custos na costura
    cols = ["ano", "n", "SR treino", "SR teste", "ex_tr", "ex_te"]
    tab = pd.DataFrame(linhas, columns=cols)
    return r.iloc[TREINO:], tab, costura


if __name__ == "__main__":
    rolante, tab, _ = walk_forward()
    ancorada, tab_a, _ = walk_forward(ancorada=True)
    print(tab.iloc[:, :4].round(2).to_string(index=False))
    print(f"médias: SR treino {tab['SR treino'].mean():.2f}, "
          f"SR teste {tab['SR teste'].mean():.2f}")
    wfe = tab["ex_te"].mean() / tab["ex_tr"].mean()
    print(f"excesso anual médio: treino "
          f"{tab['ex_tr'].mean():.1%}, teste "
          f"{tab['ex_te'].mean():.1%}; razão {wfe:.0%}")

    fora = rets.iloc[TREINO:]               # mesmo período
    sr = fora.apply(sharpe)
    print(f"melhor n fixo no período: {sr.idxmax()} "
          f"(Sharpe {sr.max():.2f}); mediana da grade: "
          f"{sr.median():.2f}")
    print(tabela({"n fixo ótimo": fora[sr.idxmax()],
                  "WF rolante": rolante,
                  "WF ancorada": ancorada}, cdi))
    print("n na ancorada:", tab_a["n"].tolist())
