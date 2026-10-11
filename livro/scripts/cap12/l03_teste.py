import numpy as np
import pandas as pd
from scipy import stats

from projeto import (ACOES, TESTE, TREINO, alvo_fixo,
                     alvo_momentum, ativo_mensal, carregar,
                     metricas, operacoes, simular)

J, K = 12, 2     # fixados em l02_selecao.py, só com o treino
px, ret, cdi = carregar()
carteiras = {"momentum": alvo_momentum(px, J, K),
             "pesos iguais": alvo_fixo(px, ACOES),
             "BOVA11": alvo_fixo(px, ["BOVA11.SA"])}

for nome, periodo in [("TREINO", TREINO), ("TESTE", TESTE)]:
    tabela, serie = {}, {}
    for c, alvo in carteiras.items():
        r, pos, giro = simular(alvo, ret, cdi, periodo)
        m = metricas(r, cdi)
        ops = operacoes(pos, px)
        m["giro anual"] = giro.sum() / (len(r) / 252)
        m["operações"] = len(ops)
        m["taxa de acerto"] = (ops["retorno"] > 0).mean()
        tabela[c], serie[c] = m, r
    print(nome, periodo[0][:4], "a", periodo[1][:4])
    print(pd.DataFrame(tabela).round(3).to_string())
    a, ri, t = ativo_mensal(serie["momentum"],
                            serie["pesos iguais"])
    ep = a.std() / np.sqrt(len(a))
    print(f"ativo contra pesos iguais, {len(a)} meses: média "
          f"{a.mean():.2%}, desvio {a.std():.2%}, erro-padrão "
          f"{ep:.2%}")
    p = 2 * stats.t.sf(abs(t), len(a) - 1)
    print(f"meses com retorno ativo positivo: {(a > 0).sum()}; "
          f"valor-p bilateral de t: {p:.2f}")
    print(f"RI = {ri:.2f}, t = {t:.2f}, IC 95% da média: "
          f"{a.mean() - 1.96 * ep:+.1%} a "
          f"{a.mean() + 1.96 * ep:+.1%}")
    n = len(r)
    taxa = (1 + cdi.loc[r.index]).prod() ** (252 / n) - 1
    sr = tabela["momentum"]["Sharpe"]
    ep = np.sqrt((1 + sr**2 / 504) * 252 / n)   # Lo (2002), iid
    print(f"CDI: {taxa:.2%} a.a.; erro-padrão do Sharpe do "
          f"momentum: {ep:.2f}; pregões: {n}\n")
