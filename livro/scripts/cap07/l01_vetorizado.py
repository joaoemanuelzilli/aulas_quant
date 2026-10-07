from motor import carregar, cruzamento

preco, ret = carregar("PETR4.SA")
bt = cruzamento(preco, ret["cdi"], rapida=20, lenta=100,
                custo=0.001)
bt, ret = bt.loc["2011":"2025"], ret.loc["2011":"2025"]

capital = (1 + bt["retorno"]).cumprod()
print(f"pregões: {len(bt)}")
print(f"tempo comprado: {bt['posicao'].mean():.1%}")
print(f"ordens: {int(bt['giro'].sum())}   "
      f"custo total: {0.001 * bt['giro'].sum():.2%}")
print(f"R$ 1 na estratégia vira  {capital.iloc[-1]:.2f}")
for nome in ["ativo", "cdi"]:
    final = (1 + ret[nome]).prod()
    print(f"R$ 1 em {nome:5s} vira       {final:.2f}")

# teste da hipótese de execução: um pregão de atraso
lento = cruzamento(preco, ret["cdi"], defasagem=2)
final = (1 + lento["retorno"].loc["2011":"2025"]).prod()
print(f"com um pregão de atraso: {final:.2f}")
