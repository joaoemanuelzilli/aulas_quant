"""O melhor Sharpe entre N estratégias sem habilidade.

Cada estratégia é ruído puro: retornos diários normais de média
zero. Escolhe-se a de maior Sharpe em 5 anos e mede-se a mesma
estratégia nos 5 anos seguintes.
"""
import numpy as np
from scipy.stats import norm

DIAS, ANOS = 252, 5
T = DIAS * ANOS
GAMA = 0.5772156649015329        # constante de Euler-Mascheroni


def sharpe(x):
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(DIAS)


def maximo_esperado(n, desvio=1.0):
    """Valor esperado do maior de n Sharpes sem habilidade."""
    if n == 1:
        return 0.0
    z1 = norm.ppf(1 - 1 / n)
    z2 = norm.ppf(1 - 1 / (n * np.e))
    return desvio * ((1 - GAMA) * z1 + GAMA * z2)


def experimento(n, repeticoes=1000, semente=2026):
    rng = np.random.default_rng(semente + n)
    dentro, fora = np.empty(repeticoes), np.empty(repeticoes)
    for k in range(repeticoes):
        r = rng.normal(0.0, 0.01, (n, 2 * T))
        s_dentro = sharpe(r[:, :T])
        melhor = s_dentro.argmax()            # escolhida antes
        dentro[k] = s_dentro[melhor]
        fora[k] = sharpe(r[[melhor], T:])[0]  # e levada adiante
    return dentro, fora


if __name__ == "__main__":
    import pandas as pd
    print("   N  dentro(média)  teoria   fora(média)")
    tabela = {}
    for n in [1, 10, 100, 1000]:
        dentro, fora = experimento(n)
        tabela[f"dentro_{n}"], tabela[f"fora_{n}"] = dentro, fora
        teoria = maximo_esperado(n, 1 / np.sqrt(ANOS))
        print(f"{n:4d}  {dentro.mean():12.2f}  {teoria:6.2f}  "
              f"{fora.mean():11.2f}")
    # guarda os resultados para a figura (a simulação é demorada)
    destino = "../../dados/c07_sobreajuste.csv"
    pd.DataFrame(tabela).to_csv(destino, index=False)
    # quantas tentativas bastam para um Sharpe 1 em 5 anos?
    n = 2
    while maximo_esperado(n, 1 / np.sqrt(ANOS)) < 1.0:
        n += 1
    print(f"Sharpe 1.0 esperado em {ANOS} anos com N = {n}")
