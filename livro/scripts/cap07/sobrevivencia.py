"""Simulação do viés de sobrevivência.

500 ações fictícias, todas com a mesma distribuição de retornos.
Uma ação é deslistada quando perde 80% do valor inicial; quem a
tinha em carteira fica com o que sobrou, parado em caixa.
"""
import numpy as np

ANOS, N, DIAS = 15, 500, 252
MU, SIGMA, BARREIRA = 0.10, 0.40, 0.20


def simular(semente=11):
    rng = np.random.default_rng(semente)
    passos = rng.normal((MU - SIGMA ** 2 / 2) / DIAS,
                        SIGMA / np.sqrt(DIAS), (ANOS * DIAS, N))
    valor = np.exp(passos.cumsum(axis=0))    # começa perto de 1
    caiu = valor < BARREIRA
    morta = np.maximum.accumulate(caiu, axis=0)  # não ressuscita
    dia_morte = morta.argmax(axis=0)
    for j in np.flatnonzero(morta[-1]):      # congela o valor
        valor[dia_morte[j]:, j] = valor[dia_morte[j], j]
    return valor, morta[-1]


if __name__ == "__main__":
    valor, morta = simular()
    viva = ~morta
    todas = valor.mean(axis=1)               # carteira completa
    vivas = valor[:, viva].mean(axis=1)      # só sobreviventes
    anual = lambda v: v[-1] ** (1 / ANOS) - 1
    print(f"deslistadas: {morta.sum()} de {N} "
          f"({morta.mean():.1%})")
    print(f"universo completo:  R$ 1 vira {todas[-1]:.2f}  "
          f"({anual(todas):.1%} a.a.)")
    print(f"só sobreviventes:   R$ 1 vira {vivas[-1]:.2f}  "
          f"({anual(vivas):.1%} a.a.)")
    print(f"viés: {anual(vivas) - anual(todas):.1%} ao ano")

    # regra: no fim do ano 5, comprar as 50 que mais caíram
    t0 = 5 * DIAS
    caiu = np.maximum.accumulate(valor < BARREIRA, axis=0)
    universos = {"completo": ~caiu[t0], "sobreviventes": viva}
    for nome, disponivel in universos.items():
        idx = np.flatnonzero(disponivel)
        sel = idx[np.argsort(valor[t0, idx])][:50]
        ganho = (valor[-1, sel] / valor[t0, sel]).mean()
        print(f"perdedoras, universo {nome}: "
              f"{ganho ** (1 / 10) - 1:.1%} a.a.; "
              f"{morta[sel].sum()} das 50 deslistadas depois")
