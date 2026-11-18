import pandas as pd
from motor import carregar, backtest, tabela, operacoes
from regras import tendencia

p, ret, cdi = carregar("BOVA11.SA")
p, ret = p["BOVA11.SA"], ret["BOVA11.SA"]

alvo = tendencia(p, n=200)       # 1 acima da média, 0 abaixo
sempre = pd.Series(1.0, index=p.index)
bruto, giro = backtest(alvo, ret, cdi, custo=0)
liquido, _ = backtest(alvo, ret, cdi, custo=0.001)
comprado, _ = backtest(sempre, ret, cdi, custo=0)

print(tabela({"BOVA11": comprado, "regra s/ custo": bruto,
              "regra líquida": liquido}, cdi))
print(f"tempo comprado: {alvo.mean():.0%}   giro anual: "
      f"{giro.sum() / len(giro) * 252:.1f}   CDI a.a.: "
      f"{(1 + cdi).prod() ** (252 / len(cdi)) - 1:.1%}")
mensal = (1 + liquido).resample("ME").prod() - 1
bova_m = (1 + comprado).resample("ME").prod() - 1
print(f"assimetria mensal: regra {mensal.skew():+.2f}   "
      f"BOVA11 {bova_m.skew():+.2f}")
ops = operacoes(alvo, ret, cdi)   # resultado acima do CDI
ganhos, perdas = ops[ops > 0], ops[ops <= 0]
print(f"{len(ops)} operações, {len(ganhos) / len(ops):.0%} com "
      f"ganho; ganho médio {ganhos.mean():+.1%}, perda média "
      f"{perdas.mean():+.1%}")
