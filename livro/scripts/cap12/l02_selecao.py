from projeto import *

px, ret, cdi = carregar()
sharpe = pd.DataFrame(index=GRADE_J, columns=GRADE_K, dtype=float)
for j in GRADE_J:
    for k in GRADE_K:
        r, _, _ = simular(alvo_momentum(px, j, k), ret, TREINO)
        sharpe.loc[j, k] = metricas(r, cdi)["Sharpe"]

sharpe.index.name, sharpe.columns.name = "J", "K"
print("Sharpe líquido no treino (2013 a 2019)")
print(sharpe.round(2))
j, k = sharpe.stack().idxmax()
print(f"escolha: J = {j}, K = {k}")
print(f"mediana da grade: {sharpe.stack().median():.2f}")
