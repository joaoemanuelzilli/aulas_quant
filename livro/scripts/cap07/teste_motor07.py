"""Testes de sanidade do motor do Capítulo 7 (cap07/motor.py)."""
import numpy as np
import pandas as pd

from motor import carregar, cruzamento

preco, ret = carregar("PETR4.SA")
cdi = ret["cdi"]


def teste_sem_sinal_rende_o_cdi():
    # médias impossíveis de cruzar: lenta maior que a amostra
    bt = cruzamento(preco, cdi, rapida=20, lenta=len(preco) + 1)
    assert (bt["posicao"] == 0).all()
    assert np.allclose(bt["retorno"], cdi)


def teste_custo_e_custo_vezes_giro():
    com = cruzamento(preco, cdi, custo=0.001)
    sem = cruzamento(preco, cdi, custo=0.0)
    cobrado = sem["retorno"] - com["retorno"]
    assert np.allclose(cobrado, 0.001 * com["giro"])
    assert set(com["giro"].unique()) <= {0.0, 1.0}


def teste_o_futuro_nao_muda_o_passado():
    # cortar a amostra em qualquer dia não altera o que veio antes
    cheio = cruzamento(preco, cdi)
    for corte in (300, 1500, 3000):
        parte = cruzamento(preco.iloc[:corte], cdi.iloc[:corte])
        assert parte.equals(cheio.iloc[:corte])


def teste_defasagem_zero_e_melhor_que_um():
    # regra que compra na alta: com o sinal do próprio dia rende
    # mais; se rende igual, a defasagem não está sendo aplicada
    final = [(1 + cruzamento(preco, cdi, 5, 20, 0.0,
                             defasagem=d)["retorno"]).prod()
             for d in (0, 1)]
    assert final[0] > final[1]


def teste_laco_explicito_da_o_mesmo():
    P, C = preco.to_numpy(), cdi.to_numpy()
    cheio = cruzamento(preco, cdi)["retorno"].to_numpy()
    ontem, sinal = 0.0, 0.0           # posição e sinal da véspera
    for t in range(len(P)):
        pos = sinal                   # decidido no fechamento t-1
        r = P[t] / P[t - 1] - 1 if t else 0.0
        liq = pos * r + (1 - pos) * C[t]
        liq -= 0.001 * abs(pos - ontem)
        assert abs(liq - cheio[t]) < 1e-12, preco.index[t]
        ontem = pos
        if t >= 99:                   # sinal com dados até t
            sinal = float(P[t - 19:t + 1].mean()
                          > P[t - 99:t + 1].mean())


if __name__ == "__main__":
    testes = [f for n, f in sorted(globals().items())
              if n.startswith("teste_")]
    for f in testes:
        f()
    print(f"{len(testes)} testes passaram")
