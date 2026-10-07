import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
m, n = 200, 1260           # 200 "estratégias", 5 anos de pregões
ruido = rng.normal(0.0, 0.01, size=(n, m))    # média real: 0

t = ruido.mean(axis=0) / (ruido.std(axis=0, ddof=1) / np.sqrt(n))
p = 2 * stats.t.sf(np.abs(t), df=n - 1)

print("p < 5%:", (p < 0.05).sum(), "de", m)
print(f"maior |t|: {np.abs(t).max():.2f}"
      f"   Sharpe anual da 'melhor': "
      f"{np.abs(t).max() / np.sqrt(n / 252):.2f}")
print(f"P(ao menos um falso positivo) = {1 - 0.95 ** m:.5f}")
alfa_b = 0.05 / m                 # correção de Bonferroni
print(f"Bonferroni: alfa = {alfa_b:.5f}"
      f"   |t| crítico = {stats.norm.isf(alfa_b / 2):.2f}"
      f"   rejeições = {(p < alfa_b).sum()}")
