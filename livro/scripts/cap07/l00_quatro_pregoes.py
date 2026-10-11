"""O exemplo de quatro pregões e o exercício da defasagem."""
import numpy as np


def estrategia(sinal, ativo, cdi, custo, defasagem):
    """Retorno diário: Equação do retorno da estratégia."""
    w = np.concatenate([np.zeros(defasagem), sinal])[:len(sinal)]
    giro = np.abs(np.diff(np.concatenate([[0.0], w])))
    return w * ativo + (1 - w) * cdi - custo * giro


exemplo = dict(sinal=np.array([1.0, 1, 0, 0]),
               ativo=np.array([0.01, 0.02, -0.01, 0.03]),
               cdi=0.0004, custo=0.001)
exercicio = dict(sinal=np.array([0.0, 1, 1, 1, 0]),
                 ativo=np.array([-1, 2, 1, -1, -2]) / 100,
                 cdi=0.0005, custo=0.001)
casos = [("exemplo", exemplo), ("exercício", exercicio)]
for nome, caso in casos:
    for defasagem in [1, 0]:
        r = estrategia(**caso, defasagem=defasagem)
        if nome == "exercício":
            r = r[1:]                    # de terça a sexta
        diarios = "  ".join(f"{x:+.2%}" for x in r)
        print(f"{nome}, defasagem {defasagem}: {diarios}   "
              f"acumulado {np.prod(1 + r) - 1:+.2%}")
