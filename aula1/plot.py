import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

precos = pd.Series([100, 102, 90, 96, 150, 158, 190, 307, 280, 299])
print(precos.to_string())

retorno_simples = precos.pct_change()

log_retorno = np.log(precos / precos.shift(1))

plt.plot(retorno_simples.index, retorno_simples.values, label='Retorno Simples')
plt.plot(log_retorno.index, log_retorno.values, label='Log Retorno')
plt.legend()

#salva o gráfico em outputs/retornos.png
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "retornos.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Gráfico salvo em: {output_file}")

