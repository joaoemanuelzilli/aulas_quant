import numpy as np
import pandas as pd

from calculo import gradiente_descendente

px = pd.read_csv("../../dados/precos_ajustados.csv",
                 index_col=0, parse_dates=True)
px = px[["^BVSP", "VALE3.SA"]].loc["2023-12":"2024"].dropna()
r = 100 * px.pct_change().dropna().loc["2024"]  # em %
x, y = r["^BVSP"].to_numpy(), r["VALE3.SA"].to_numpy()


def grad(t):
    """Gradiente de L(a, b) = média de (y - a - b x)^2."""
    e = y - t[0] - t[1] * x
    return np.array([-2 * e.mean(), -2 * (e * x).mean()])


b_mqo, a_mqo = np.polyfit(x, y, 1)   # solução fechada
print(f"n = {len(x)}  fechada: a = {a_mqo:.4f}  b = {b_mqo:.4f}")
for eta in [0.05, 0.5, 0.95, 1.05]:
    c = gradiente_descendente(grad, [0.6, 0], eta, 40)
    a, b = c[-1]
    print(f"eta = {eta:<4}  40 passos: a = {a:.4f}  b = {b:.4f}")
