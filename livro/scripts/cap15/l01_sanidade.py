from projeto import CUSTO, TESTE, alvo_fixo, carregar, simular

px, ret, cdi = carregar()
bova = alvo_fixo(px, ["BOVA11.SA"])
r, pos, giro = simular(bova, ret, cdi, TESTE)

# conta direta: um dia de CDI em caixa, compra no fechamento
# do primeiro pregão, um custo de entrada, e segura até o fim
p = px.loc[TESTE[0]:TESTE[1], "BOVA11.SA"]
c0 = cdi.loc[p.index[0]]
direto = (1 + c0) * (1 - CUSTO) * p.iloc[-1] / p.iloc[0] - 1

print(f"simulador:    {(1 + r).prod() - 1:.6f}")
print(f"conta direta: {direto:.6f}")
print(f"giro total:   {giro.sum():.4f}")
print(f"primeira ordem em {giro[giro > 0].index[0].date()}, "
      f"sinal de {bova.loc[:TESTE[0]].index[-1].date()}")

# fora do mercado o tempo todo: o resultado tem de ser o CDI
r0, _, _ = simular(bova * 0, ret, cdi, TESTE)
so_cdi = (1 + cdi.loc[TESTE[0]:TESTE[1]]).prod() - 1
print(f"sempre em caixa: {(1 + r0).prod() - 1:.6f}   "
      f"CDI: {so_cdi:.6f}")
