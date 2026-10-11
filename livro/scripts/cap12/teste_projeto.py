"""Testes de sanidade do projeto-exemplo (cap12/projeto.py)."""
import numpy as np
import pandas as pd

from projeto import (ACOES, TESTE, alvo_fixo, alvo_momentum,
                     carregar, drawdown, simular)

px, ret, cdi = carregar()


def teste_sinal_nao_usa_o_futuro():
    cheio = alvo_momentum(px, 12, 2)
    for corte in ("2016-06-30", "2019-12-30", "2023-03-31"):
        parte = alvo_momentum(px.loc[:corte], 12, 2)
        assert parte.equals(cheio.loc[:corte]), corte


def teste_pesos_somam_um_e_sao_k_papeis():
    for k in (2, 3, 4):
        alvo = alvo_momentum(px, 6, k)
        assert np.allclose(alvo.sum(axis=1), 1.0)
        assert ((alvo > 0).sum(axis=1) == k).all()


def teste_custo_zero_e_atraso_nao_criam_dinheiro():
    alvo = alvo_fixo(px, ACOES)
    caro = simular(alvo, ret, cdi, TESTE, custo=0.005)[0]
    gratis = simular(alvo, ret, cdi, TESTE, custo=0.0)[0]
    assert (gratis >= caro - 1e-15).all()
    assert (1 + gratis).prod() > (1 + caro).prod()


def teste_posicoes_somam_um_depois_da_primeira_ordem():
    _, pos, giro = simular(alvo_momentum(px, 12, 2), ret, cdi,
                             TESTE)
    depois = pos.loc[giro[giro > 0].index[0]:]
    assert np.allclose(depois.sum(axis=1), 1.0)
    assert (pos >= 0).all().all()         # só comprado


def teste_todo_atraso_compra_no_primeiro_pregao():
    # sem isso, com atraso 0 a ordem do último sinal anterior
    # ao período se perde e a carteira fica um mês vazia
    alvo = alvo_momentum(px, 12, 2)
    for atraso in (0, 1, 2, 5):
        _, pos, _ = simular(alvo, ret, cdi, TESTE, atraso=atraso)
        assert pos.iloc[0].sum() > 0, f"atraso={atraso}"


def teste_fora_do_mercado_rende_o_cdi():
    zero = alvo_fixo(px, ACOES) * 0
    r, _, _ = simular(zero, ret, cdi, TESTE)
    assert np.allclose(r, cdi.loc[TESTE[0]:TESTE[1]])


def teste_drawdown_conta_o_capital_inicial():
    r = pd.Series([-0.10, 0.05, 0.02])
    assert np.isclose(drawdown(r).iloc[0], -0.10)


if __name__ == "__main__":
    testes = [f for n, f in sorted(globals().items())
              if n.startswith("teste_")]
    falhas = 0
    for f in testes:
        try:
            f()
        except AssertionError as e:
            falhas += 1
            print("FALHOU:", f.__name__, e)
    print(f"{len(testes) - falhas} de {len(testes)} passaram")
