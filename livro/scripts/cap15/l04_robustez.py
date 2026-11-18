from itertools import combinations

import pandas as pd

from projeto import (ACOES, TESTE, alvo_fixo, alvo_momentum,
                     ativo_mensal, carregar, metricas, simular)

J, K = 12, 2
px, ret, cdi = carregar()
ref = alvo_fixo(px, ACOES)
mom = alvo_momentum(px, J, K)


def linha(rotulo, alvo, base, **kw):
    """Sharpe da estratégia e da referência; t do ativo."""
    r, _, _ = simular(alvo, ret, cdi, TESTE, **kw)
    b, _, _ = simular(base, ret, cdi, TESTE, **kw)
    s = [metricas(x, cdi)["Sharpe"] for x in (r, b)]
    t = ativo_mensal(r, b)[2]
    print(f"{rotulo:<20}{s[0]:7.2f}{s[1]:8.2f}{t:7.2f}")
    return s[0], t


print("teste, 2020 a 2025     mom.  iguais      t")
for c in [0.0, 0.001, 0.002, 0.005]:
    linha(f"custo de {c:.2%}", mom, ref, custo=c)
for d in [0, 1, 2, 5]:
    linha(f"atraso de {d} dia(s)", mom, ref, atraso=d)
for fora in ACOES:
    resto = [a for a in ACOES if a != fora]
    linha(f"sem {fora[:-3]}", alvo_momentum(px, J, K, resto),
          alvo_fixo(px, resto))

r, _, _ = simular(mom, ret, cdi, TESTE)
b, _, _ = simular(ref, ret, cdi, TESTE)
ano = lambda s: (1 + s).groupby(s.index.year).prod() - 1
por_ano = pd.DataFrame({"momentum": ano(r), "iguais": ano(b)})
por_ano["ativo"] = por_ano["momentum"] - por_ano["iguais"]
por_ano.index.name = None
print(por_ano.map(lambda v: f"{v:+.1%}").T.to_string())

t_cheio = ativo_mensal(r, b)[2]
t_2021 = ativo_mensal(r.loc["2021":], b.loc["2021":])[2]
print(f"t do retorno ativo sem o ano de 2020: {t_2021:.2f}")

# pares fixos de duas ações, sem regra, repostos todo mês
melhores = 0
for par in combinations(ACOES, 2):
    p, _, _ = simular(alvo_fixo(px, list(par)), ret, cdi, TESTE)
    melhores += ativo_mensal(p, b)[2] >= t_cheio
print(f"pares fixos com t maior ou igual ao da estratégia: "
      f"{melhores} de 21")
