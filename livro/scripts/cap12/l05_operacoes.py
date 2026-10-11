from projeto import (TESTE, alvo_momentum, carregar, drawdown,
                     operacoes, simular)

px, ret, cdi = carregar()
r, pos, giro = simular(alvo_momentum(px, 12, 2), ret, cdi, TESTE)
ops = operacoes(pos, px)
ops["ativo"] = ops["ativo"].str[:-3]
ops["dias"] = (ops["saida"] - ops["entrada"]).dt.days
g = ops["retorno"]

print(f"operações: {len(ops)}, com ganho: {(g > 0).sum()}")
print(f"ganho médio: {g[g > 0].mean():.1%}   "
      f"perda média: {g[g <= 0].mean():.1%}")
print(f"duração mediana: {ops['dias'].median():.0f} dias")
maiores = ops.nlargest(3, "retorno")
print(maiores.to_string(index=False, formatters={
    "entrada": "{:%Y-%m-%d}".format,
    "saida": "{:%Y-%m-%d}".format,
    "retorno": "{:+.1%}".format}))
dd = drawdown(r)
print(f"último topo em {dd[dd == 0].index[-1].date()}; "
      f"fim da amostra a {dd.iloc[-1]:.1%} dele")
