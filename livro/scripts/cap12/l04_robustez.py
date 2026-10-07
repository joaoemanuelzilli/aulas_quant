from projeto import *

J, K = 12, 2
px, ret, cdi = carregar()
ref = alvo_fixo(px, ACOES)


def linha(rotulo, alvo, base, **kw):
    """Sharpe da estratégia e da referência; t do ativo."""
    r, _, _ = simular(alvo, ret, TESTE, **kw)
    b, _, _ = simular(base, ret, TESTE, **kw)
    s = [metricas(x, cdi)["Sharpe"] for x in (r, b)]
    t = ativo_mensal(r, b)[2]
    print(f"{rotulo:<20}{s[0]:7.2f}{s[1]:8.2f}{t:7.2f}")


print("teste, 2020 a 2025     mom.  iguais      t")
for c in [0.0, 0.001, 0.002, 0.005]:
    linha(f"custo de {c:.2%}", alvo_momentum(px, J, K), ref,
          custo=c)
for d in [0, 1, 2, 5]:
    linha(f"atraso de {d} dia(s)", alvo_momentum(px, J, K), ref,
          atraso=d)
for fora in ACOES:
    resto = [a for a in ACOES if a != fora]
    linha(f"sem {fora[:-3]}", alvo_momentum(px, J, K, resto),
          alvo_fixo(px, resto))

r, _, _ = simular(alvo_momentum(px, J, K), ret, TESTE)
b, _, _ = simular(ref, ret, TESTE)
ano = lambda s: (1 + s).groupby(s.index.year).prod() - 1
por_ano = pd.DataFrame({"momentum": ano(r), "iguais": ano(b)})
por_ano["ativo"] = por_ano["momentum"] - por_ano["iguais"]
por_ano.index.name = None
print(por_ano.map(lambda v: f"{v:+.1%}").T.to_string())
