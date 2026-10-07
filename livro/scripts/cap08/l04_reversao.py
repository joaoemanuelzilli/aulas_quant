import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from motor import carregar, backtest, tabela, operacoes
from regras import zscore, reversao, meia_vida

p, ret, cdi = carregar("BOVA11.SA")
p, ret = p["BOVA11.SA"], ret["BOVA11.SA"]

x = np.log(p)
desvio = (x - x.rolling(20).mean()).dropna()
print(f"ADF, log-preço: p = {adfuller(x)[1]:.3f}")
print(f"ADF, desvio à média de 20 dias: p = "
      f"{adfuller(desvio)[1]:.1e}")
h, phi = meia_vida(desvio)
print(f"AR(1) do desvio: phi = {phi:.3f}, meia-vida = "
      f"{h:.1f} dias")

# o mesmo exercício em passeios aleatórios simulados
rng = np.random.default_rng(8)
sim = []
for _ in range(500):
    w = pd.Series(rng.normal(0, 0.015, len(p)).cumsum())
    sim.append(meia_vida(w - w.rolling(20).mean())[0])
q5, q50, q95 = np.percentile(sim, [5, 50, 95])
print(f"passeio aleatório: meia-vida mediana = {q50:.1f} dias "
      f"(5% a 95%: {q5:.1f} a {q95:.1f})")

alvo = reversao(p, n=20, entrada=2.0)
bruto, giro = backtest(alvo, ret, cdi, custo=0)
liquido, _ = backtest(alvo, ret, cdi, custo=0.001)
print(tabela({"sem custo": bruto, "líquida": liquido}, cdi))
ops = operacoes(alvo, ret, cdi)
mensal = (1 + liquido).resample("ME").prod() - 1
ganhos, perdas = ops[ops > 0], ops[ops <= 0]
print(f"{len(ops)} operações, {len(ganhos) / len(ops):.0%} com "
      f"ganho; ganho médio {ganhos.mean():+.1%}, perda média "
      f"{perdas.mean():+.1%}")
print(f"pior operação: {ops.min():+.1%}   tempo comprado: "
      f"{alvo.mean():.0%}   assimetria mensal: "
      f"{mensal.skew():+.2f}")
