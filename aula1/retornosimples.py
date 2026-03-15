import pandas as pd

precos = pd.Series([100, 102, 90, 96, 150, 158, 190, 307, 280, 299])
print(precos.to_string())

retorno_simples = precos.pct_change()

print(retorno_simples.to_string())