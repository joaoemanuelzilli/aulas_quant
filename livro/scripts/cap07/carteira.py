import numpy as np
import pandas as pd

ATIVOS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
          "ABEV3.SA"]
V = 1_000_000                  # valor da carteira, em reais
w = np.full(5, 0.2)            # pesos iguais

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
p = precos[ATIVOS].dropna()    # pregões da B3
R = p.pct_change().dropna()    # retornos simples diários
Rp = R @ w                     # retorno simples da carteira
perda = -V * Rp                # perda diária em reais
cota = (1 + Rp).cumprod()      # valor de R$ 1 aplicado


def var_historico(perda, janela=250, nivel=0.99):
    """VaR histórico previsto na véspera: quantil das `janela`
    perdas anteriores, sem a perda do próprio dia."""
    return perda.rolling(janela).quantile(nivel).shift(1)


if __name__ == "__main__":
    L = perda["2015":"2024"]
    print(f"{len(L)} dias, de {L.index[0]:%d/%m/%Y} "
          f"a {L.index[-1]:%d/%m/%Y}")
    print(f"perda média: R$ {L.mean():,.0f}")
    print(f"desvio-padrão: R$ {L.std():,.0f}")
    print(f"maior perda: R$ {L.max():,.0f} "
          f"em {L.idxmax():%d/%m/%Y}")
