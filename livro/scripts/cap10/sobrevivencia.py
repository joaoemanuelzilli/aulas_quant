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


def medir(semente):
    """Carteira e regra das perdedoras nos dois universos."""
    valor, morta = simular(semente)
    viva = ~morta
    anual = lambda v: v ** (1 / ANOS) - 1
    todas = valor.mean(axis=1)[-1]           # carteira completa
    vivas = valor[:, viva].mean(axis=1)[-1]  # só sobreviventes
    # regra: no fim do ano 5, comprar as 50 que mais caíram
    t0 = 5 * DIAS
    caiu = np.maximum.accumulate(valor < BARREIRA, axis=0)
    perdedoras = {}
    for nome, disponivel in [("completo", ~caiu[t0]),
                             ("sobreviventes", viva)]:
        idx = np.flatnonzero(disponivel)
        sel = idx[np.argsort(valor[t0, idx])][:50]
        ganho = (valor[-1, sel] / valor[t0, sel]).mean()
        perdedoras[nome] = (ganho ** (1 / 10) - 1,
                            morta[sel].sum())
    return morta, todas, vivas, anual, perdedoras


if __name__ == "__main__":
    morta, todas, vivas, anual, perdedoras = medir(11)
    print(f"deslistadas: {morta.sum()} de {N} "
          f"({morta.mean():.1%})")
    print(f"universo completo:  R$ 1 vira {todas:.2f}  "
          f"({anual(todas):.1%} a.a.)")
    print(f"só sobreviventes:   R$ 1 vira {vivas:.2f}  "
          f"({anual(vivas):.1%} a.a.)")
    print(f"viés: {anual(vivas) - anual(todas):.1%} ao ano")
    for nome, (taxa, mortas) in perdedoras.items():
        print(f"perdedoras, universo {nome}: {taxa:.1%} a.a.; "
              f"{mortas} das 50 deslistadas depois")

    # o mesmo em 40 sementes: o que é estável e o que é ruído
    vies, vantagem = [], []
    for semente in range(40):
        _, todas, vivas, anual, perdedoras = medir(semente)
        vies.append(anual(vivas) - anual(todas))
        vantagem.append(perdedoras["sobreviventes"][0]
                        - perdedoras["completo"][0])
    print(f"40 sementes, viés: de {min(vies):.1%} a "
          f"{max(vies):.1%}")
    print(f"40 sementes, vantagem das perdedoras: de "
          f"{min(vantagem):.1%} a {max(vantagem):.1%}")
