import numpy as np
import pandas as pd

ATIVOS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
          "ABEV3.SA"]
V = 1_000_000                  # valor da carteira, em reais
w = np.full(5, 0.2)            # pesos iguais

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
pregoes = precos[ATIVOS].dropna(how="all")
p = pregoes.dropna()           # datas com os cinco preços
R_todos = p.pct_change().dropna()

# depois de uma data sem preço o retorno cobre vários pregões:
# ele entra na cota, mas não é uma perda diária
ordem = pd.Series(np.arange(len(pregoes)), pregoes.index)
um_pregao = ordem[p.index].diff().dropna() == 1
cota = (1 + R_todos @ w).cumprod()   # valor de R$ 1 aplicado
R = R_todos[um_pregao]
Rp = R @ w                     # retorno simples da carteira
perda = -V * Rp                # perda diária em reais

if __name__ == "__main__":
    L = perda["2015":"2024"]
    print(f"{len(L)} dias, de {L.index[0]:%d/%m/%Y} "
          f"a {L.index[-1]:%d/%m/%Y}")
    print(f"perda média: R$ {L.mean():,.0f}")
    print(f"desvio-padrão: R$ {L.std():,.0f}")
    print(f"maior perda: R$ {L.max():,.0f} "
          f"em {L.idxmax():%d/%m/%Y}")
    fora = ~um_pregao
    print(f"retornos de mais de um pregão: {fora.sum()} "
          f"({fora['2015':'2024'].sum()} de 2015 a 2024)")
