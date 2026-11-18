import timeit

import numpy as np
import pandas as pd


def medir(funcao, vezes):
    """Segundos por chamada: o menor resultado de 5 rodadas."""
    rodadas = timeit.repeat(funcao, number=vezes, repeat=5)
    return min(rodadas) / vezes


# 1. procurar um valor que não está na coleção
print("        n   lista (µs)   conjunto (µs)")
for n in (1_000, 10_000, 100_000, 1_000_000):
    lista = list(range(n))
    conjunto = set(lista)
    t_l = medir(lambda: -1 in lista, 20)
    t_c = medir(lambda: -1 in conjunto, 100_000)
    print(f"{n:9d} {t_l * 1e6:12.2f} {t_c * 1e6:15.3f}")

# 2. retornos de PETR4: laço em Python contra NumPy
petr = pd.read_csv("../../dados/precos_ajustados.csv",
                   index_col=0, parse_dates=True)["PETR4.SA"]
precos = petr.dropna().to_numpy()
lista = precos.tolist()


def com_laco(p):
    return [p[i] / p[i - 1] - 1 for i in range(1, len(p))]


def vetorizado(p):
    return p[1:] / p[:-1] - 1


assert np.allclose(com_laco(lista), vetorizado(precos))
t_laco = medir(lambda: com_laco(lista), 200)
t_np = medir(lambda: vetorizado(precos), 20_000)
print(f"{len(precos) - 1} retornos: laço {t_laco * 1e6:.0f} µs,",
      f"NumPy {t_np * 1e6:.1f} µs ({t_laco / t_np:.0f} vezes)")
