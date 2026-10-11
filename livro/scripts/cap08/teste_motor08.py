"""Testes de sanidade do motor e das regras do Capítulo 8.

Rodar de dentro de scripts/cap08:  python teste_motor08.py
"""
import numpy as np
import pandas as pd
from motor import carregar, carregar_ohlc, backtest
from regras import tendencia, reversao, donchian

ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
p, ret, cdi = carregar(*ACOES)

# 1. tudo em caixa rende o CDI, com giro zero
r, giro = backtest(p * 0.0, ret, cdi)
assert np.allclose(r, cdi) and giro.sum() == 0

# 2. um ativo, sem custo: é o próprio ativo, e o giro total é 1
um = pd.Series(1.0, index=p.index)
r, giro = backtest(um, ret["PETR4.SA"], cdi, custo=0)
assert np.isclose((1 + r.iloc[1:]).prod(), p["PETR4.SA"].iloc[-1] / p["PETR4.SA"].iloc[0])
assert np.isclose(giro.sum(), 1.0)

# 3. carteira 1/3 com custo: confere com uma contabilidade em reais
alvo = p * 0.0 + 1 / 3
r, giro = backtest(alvo, ret, cdi, custo=0.001)
cota, pos = 1.0, np.zeros(3)            # pos: reais em cada ação
for t in range(len(p)):
    caixa = cota - pos.sum()
    pos = pos * (1 + ret.iloc[t].values)
    cota = pos.sum() + caixa * (1 + cdi.iloc[t])
    novo = cota / 3 * np.ones(3)        # volta ao alvo no fechamento
    cota -= 0.001 * np.abs(novo - pos).sum()   # custo sai do caixa
    pos = novo
erro = abs((1 + r).prod() / cota - 1)
assert erro < 2e-3, erro                # o motor debita o custo um dia depois

# 4. as regras não mudam o passado quando a amostra é cortada
ohlc, _, _ = carregar_ohlc()
corte = 2500
for nome, regra, dados in [("tendencia", tendencia, p["PETR4.SA"]),
                           ("reversao", reversao, p["PETR4.SA"]),
                           ("donchian", donchian, ohlc)]:
    inteiro, cortado = regra(dados), regra(dados.iloc[:corte])
    assert inteiro.iloc[:corte].equals(cortado), nome

# 5. o motor recusa retorno ausente e alvo com outro índice
ruim = ret.copy(); ruim.iloc[10, 0] = np.nan
for args in [(alvo, ruim, cdi), (alvo.iloc[::21], ret, cdi)]:
    try:
        backtest(*args)
        raise SystemExit("o motor aceitou entrada inválida")
    except AssertionError:
        pass

print(f"5 testes passaram (diferença de cota no teste 3: {erro:.1e})")
