"""Testes das funções centrais do Capítulo 16 (algoritmos.py).

Cada versão rápida é comparada com a ingênua em centenas de
entradas aleatórias e em casos de borda. Rode de dentro de
scripts/cap16/:  python teste_algoritmos.py
"""
import bisect

import numpy as np
import pandas as pd

from algoritmos import (busca_binaria, busca_linear, dois_somam,
                        dois_somam_ingenuo, drawdown_maximo,
                        drawdown_maximo_ingenuo, media_movel,
                        media_movel_ingenua, media_movel_np,
                        melhor_negocio, melhor_negocio_ingenuo)

rng = np.random.default_rng(16)


def precos_aleatorios(n):
    return list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))))


def teste_melhor_negocio_igual_ao_ingenuo():
    for _ in range(300):
        p = precos_aleatorios(int(rng.integers(1, 60)))
        r, ri = melhor_negocio(p), melhor_negocio_ingenuo(p)
        assert np.isclose(r[0], ri[0])
        # o par devolvido realiza o lucro informado
        if r[0] > 0:
            assert r[1] < r[2]
            assert np.isclose(p[r[2]] - p[r[1]], r[0])


def teste_melhor_negocio_bordas():
    assert melhor_negocio([]) == (0.0, 0, 0)
    assert melhor_negocio([5.0]) == (0.0, 0, 0)
    assert melhor_negocio([5, 4, 3, 2])[0] == 0.0  # só cai
    assert melhor_negocio([7, 1, 5, 3, 6, 4]) == (5, 1, 4)


def teste_dois_somam():
    for _ in range(300):
        a = list(rng.integers(-20, 20, int(rng.integers(0, 30))))
        alvo = int(rng.integers(-30, 30))
        r, ri = dois_somam(a, alvo), dois_somam_ingenuo(a, alvo)
        assert (r is None) == (ri is None)
        if r is not None:
            i, j = r
            assert i < j and a[i] + a[j] == alvo
    assert dois_somam([3, 3], 6) == (0, 1)  # valor repetido
    assert dois_somam([3], 6) is None  # não usa o 3 duas vezes


def teste_drawdown_igual_ao_ingenuo_e_ao_pandas():
    for _ in range(200):
        v = precos_aleatorios(int(rng.integers(1, 80)))
        s = pd.Series(v)
        dd_pd = (s / s.cummax() - 1).min()
        assert np.isclose(drawdown_maximo(v),
                          drawdown_maximo_ingenuo(v))
        assert np.isclose(drawdown_maximo(v), dd_pd)
    assert drawdown_maximo([1, 2, 3]) == 0.0
    for f in (drawdown_maximo, drawdown_maximo_ingenuo):
        try:                      # série vazia: erro explícito
            f([])
        except (IndexError, ValueError):
            pass
        else:
            raise AssertionError("série vazia deveria falhar")
    assert np.isclose(drawdown_maximo([10, 5, 20, 8]), -0.6)


def teste_media_movel():
    for _ in range(200):
        n = int(rng.integers(1, 80))
        k = int(rng.integers(1, n + 1))
        x = list(rng.normal(0, 1, n))
        a = media_movel(x, k)
        assert np.allclose(a, media_movel_ingenua(x, k))
        assert np.allclose(a, media_movel_np(x, k))
        ref = pd.Series(x).rolling(k).mean().dropna()
        assert np.allclose(a, ref)


def teste_busca_binaria():
    for _ in range(300):
        a = sorted(rng.integers(0, 50, int(rng.integers(0, 40))))
        x = int(rng.integers(-5, 55))
        esperado = bisect.bisect_left(a, x)
        assert busca_binaria(a, x) == esperado
        assert busca_linear(a, x) == esperado
        assert np.searchsorted(a, x) == esperado


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
    if falhas:
        raise SystemExit(1)
