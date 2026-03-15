import pandas as pd
import numpy as np

precos = pd.Series([100, 102, 90, 96, 150, 158, 190, 307, 280, 299])
print(precos.to_string())

log_retorno = np.log(precos / precos.shift(1))

print(log_retorno.to_string())