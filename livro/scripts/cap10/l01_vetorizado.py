from motor import carregar, cruzamento

preco, ret = carregar("PETR4.SA")
janela = slice("2011", "2025")
bt = cruzamento(preco, ret["cdi"], rapida=20, lenta=100,
                custo=0.001).loc[janela]

capital = (1 + bt["retorno"]).cumprod()
print(f"pregões: {len(bt)}")
print(f"tempo comprado: {bt['posicao'].mean():.1%}")
print(f"ordens: {int(bt['giro'].sum())}   "
      f"custo total: {0.001 * bt['giro'].sum():.2%}")
print(f"R$ 1 na estratégia vira  {capital.iloc[-1]:.2f}")
finais = {}
for nome in ["ativo", "cdi"]:
    finais[nome] = (1 + ret.loc[janela, nome]).prod()
    print(f"R$ 1 em {nome:5s} vira       {finais[nome]:.2f}")
pct_cdi = (capital.iloc[-1] - 1) / (finais["cdi"] - 1)
print(f"ganho da estratégia em % do ganho do CDI: {pct_cdi:.0%}")

# teste da hipótese de execução: um pregão de atraso
lento = cruzamento(preco, ret["cdi"], defasagem=2)
final = (1 + lento["retorno"].loc[janela]).prod()
print(f"com um pregão de atraso: {final:.2f}")

# teste da hipótese do caixa: 80% do CDI e caixa parado
for fracao in [0.8, 0.0]:
    outro = cruzamento(preco, fracao * ret["cdi"])
    final = (1 + outro["retorno"].loc[janela]).prod()
    print(f"com o caixa a {fracao:.0%} do CDI: {final:.2f}")

# quanto tempo dura cada passagem pelo caixa
fora = bt["posicao"] == 0
duracao = fora.groupby((fora != fora.shift()).cumsum()).sum()
duracao = duracao[duracao > 0]
print(f"períodos em caixa: {len(duracao)}   "
      f"com menos de 126 pregões: {(duracao < 126).sum()}")
