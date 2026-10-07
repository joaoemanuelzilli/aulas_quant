import numpy as np

precos = np.array([20.00, 20.50, 19.90, 21.10, 22.00])

# operação vetorizada: age sobre o array inteiro, sem laço
retornos = precos[1:] / precos[:-1] - 1

print(retornos.round(4))
print(f"média: {retornos.mean():.4%}   "
      f"desvio-padrão: {retornos.std(ddof=1):.4%}")
