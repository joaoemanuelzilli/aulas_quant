import pandas as pd

from projeto import (ACOES, GRADE_J, GRADE_K, TREINO, alvo_fixo,
                     alvo_momentum, ativo_mensal, carregar,
                     metricas, simular)

px, ret, cdi = carregar()


def grade(acoes):
    """Sharpe das 12 variantes e da carteira passiva."""
    base, _, _ = simular(alvo_fixo(px, acoes), ret, cdi, TREINO)
    g = pd.DataFrame(index=GRADE_J, columns=GRADE_K, dtype=float)
    perdem = 0
    for j in GRADE_J:
        for k in GRADE_K:
            alvo = alvo_momentum(px, j, k, acoes)
            r, _, _ = simular(alvo, ret, cdi, TREINO)
            g.loc[j, k] = metricas(r, cdi)["Sharpe"]
            perdem += ativo_mensal(r, base)[1] < 0
    g.index.name, g.columns.name = "J", "K"
    return g, metricas(base, cdi)["Sharpe"], perdem


sharpe, passiva, perdem = grade(ACOES)
print("Sharpe líquido no treino (2013 a 2019)")
print(sharpe.round(2))
j, k = sharpe.stack().idxmax()
print(f"escolha: J = {j}, K = {k}")
print(f"pesos iguais: {passiva:.2f}; variantes com retorno "
      f"ativo negativo: {perdem} de 12")

resto = [a for a in ACOES if a != "MGLU3.SA"]
sem, passiva, perdem = grade(resto)
print("\nA mesma grade sem MGLU3 no universo")
print(sem.round(2))
print(f"pesos iguais: {passiva:.2f}; variantes com retorno "
      f"ativo negativo: {perdem} de 12")
