import numpy as np
import statsmodels.api as sm
rng = np.random.default_rng(42)
T, N = 500, 2000
t_nivel, t_dif, r2 = [], [], []
for _ in range(N):
    # dois passeios aleatórios sem nenhuma relação entre si
    x = rng.standard_normal(T).cumsum()
    y = rng.standard_normal(T).cumsum()
    nivel = sm.OLS(y, sm.add_constant(x)).fit()
    dx, dy = np.diff(x), np.diff(y)
    dif = sm.OLS(dy, sm.add_constant(dx)).fit()
    t_nivel.append(nivel.tvalues[1])
    t_dif.append(dif.tvalues[1])
    r2.append(nivel.rsquared)
t_nivel, t_dif, r2 = map(np.array, (t_nivel, t_dif, r2))
rej_nivel = np.mean(abs(t_nivel) > 1.96)
rej_dif = np.mean(abs(t_dif) > 1.96)
print(f"|t| > 1,96 em níveis:     {rej_nivel:.1%}")
print(f"|t| > 1,96 em diferenças: {rej_dif:.1%}")
print(f"R² acima de 0,5 em níveis: {np.mean(r2 > 0.5):.1%}")
