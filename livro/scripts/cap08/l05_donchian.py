import pandas as pd
from motor import carregar_ohlc, backtest, tabela, operacoes
from regras import donchian

ohlc, ret, cdi = carregar_ohlc()

alvo = donchian(ohlc, n=20)
sempre = pd.Series(1.0, index=ohlc.index)
liquido, giro = backtest(alvo, ret, cdi, custo=0.001)
comprado, _ = backtest(sempre, ret, cdi, custo=0)
print(tabela({"PETR4": comprado, "Donchian 20": liquido}, cdi))

ops = operacoes(alvo, ret, cdi)
ganhos, perdas = ops[ops > 0], ops[ops <= 0]
mensal = (1 + liquido).resample("ME").prod() - 1
print(f"{len(ops)} operações, {len(ganhos) / len(ops):.0%} com "
      f"ganho; ganho médio {ganhos.mean():+.1%}, perda média "
      f"{perdas.mean():+.1%}")
print(f"melhor operação: {ops.max():+.1%}   tempo comprado: "
      f"{alvo.mean():.0%}   assimetria mensal: "
      f"{mensal.skew():+.2f}")
top3 = ops.nlargest(3).sum() / ops.sum()
print(f"as 3 maiores operações somam {top3:.0%} do resultado")
