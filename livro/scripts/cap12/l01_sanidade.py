from projeto import *

px, ret, cdi = carregar()
bova = alvo_fixo(px, ["BOVA11.SA"])
r, pos, giro = simular(bova, ret, TESTE)

# conta direta: compra no fechamento do primeiro pregão do
# período, paga o custo uma vez e segura até o fim
p = px.loc[TESTE[0]:TESTE[1], "BOVA11.SA"]
direto = p.iloc[-1] / p.iloc[0] * (1 - CUSTO) - 1

print(f"simulador:    {(1 + r).prod() - 1:.6f}")
print(f"conta direta: {direto:.6f}")
print(f"giro total:   {giro.sum():.4f}")
print(f"primeira ordem em {giro[giro > 0].index[0].date()}, "
      f"sinal de {bova.loc[:TESTE[0]].index[-1].date()}")
