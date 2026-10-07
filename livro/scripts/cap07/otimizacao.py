"""Grade de parâmetros do cruzamento de médias em PETR4.

Sharpe de cada par (rápida, lenta) dentro da amostra (2011-2018)
e fora dela (2019-2025).
"""
import pandas as pd
from scipy.stats import spearmanr

import metricas as m
from motor import carregar, cruzamento

RAPIDAS = list(range(5, 55, 5))
LENTAS = list(range(20, 220, 20))
PERIODOS = {"dentro": slice("2011", "2018"),
            "fora": slice("2019", "2025")}


def grade():
    preco, ret = carregar("PETR4.SA")
    linhas = []
    for rapida in RAPIDAS:
        for lenta in LENTAS:
            if rapida >= lenta:
                continue
            bt = cruzamento(preco, ret["cdi"], rapida, lenta)
            linha = {"rapida": rapida, "lenta": lenta}
            for nome, fatia in PERIODOS.items():
                linha[nome] = m.sharpe(bt["retorno"].loc[fatia],
                                       ret["cdi"].loc[fatia])
            linhas.append(linha)
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    g = grade().sort_values("dentro", ascending=False)
    print(f"combinações testadas: {len(g)}")
    print(g.head(3).round(2).to_string(index=False))
    print("mediana dentro:", round(g["dentro"].median(), 2),
          "  mediana fora:", round(g["fora"].median(), 2))
    posto = g["fora"].rank(ascending=False).iloc[0]
    print(f"a campeã de dentro é a {int(posto)}ª fora da amostra")
    rho = spearmanr(g["dentro"], g["fora"])[0]
    print(f"correlação de postos dentro x fora: {rho:.2f}")
    # o que N tentativas sem habilidade dariam em 8 anos
    from sobreajuste import maximo_esperado
    for n in [10, 90]:
        esperado = maximo_esperado(n, 1 / 8 ** 0.5)
        print(f"máximo esperado sem habilidade, N = {n}: "
              f"{esperado:.2f}")
    preco, ret = carregar("PETR4.SA")
    for nome, fatia in PERIODOS.items():
        r, rf = ret["ativo"].loc[fatia], ret["cdi"].loc[fatia]
        s = m.sharpe(r, rf)
        print(f"Sharpe de comprar e segurar, {nome}: {s:.2f}")
