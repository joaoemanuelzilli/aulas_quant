"""Testes das funções de l11_significancia.py."""
import numpy as np
import pandas as pd

from l11_significancia import (bootstrap_blocos, retorno_do_sinal,
                               t_media)
from motor import carregar, cruzamento


def teste_t_iid_igual_sharpe_vezes_raiz():
    rng = np.random.default_rng(0)
    d = rng.normal(0.001, 0.01, 2520)
    sr = d.mean() / d.std(ddof=1)
    assert np.isclose(t_media(d), sr * np.sqrt(len(d)))


def teste_newey_west_sem_defasagem_e_quase_iid():
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 0.01, 5000)
    # HAC com 0 defasagens = erro-padrão robusto de White
    assert abs(t_media(d, 0) / t_media(d) - 1) < 0.01


def teste_newey_west_cresce_com_autocorrelacao():
    rng = np.random.default_rng(2)
    e = rng.normal(0.0005, 0.01, 6000)
    d = e[1:] + 0.8 * e[:-1]          # MA(1) positivo
    assert t_media(d, 10) < 0.8 * t_media(d)


def teste_bootstrap_bloco_um_e_media_de_dias():
    rng = np.random.default_rng(3)
    d = rng.normal(0.0005, 0.01, 1000)
    s = bootstrap_blocos(d, 1, 3000, rng)
    ep = 1 / np.sqrt(len(d) / 252)     # Eq. de Lo, Sharpe pequeno
    assert abs(s.std() / ep - 1) < 0.1


def teste_sinal_reproduz_motor():
    preco, ret = carregar("PETR4.SA")
    r_a = preco.pct_change().fillna(0.0)
    s = (preco.rolling(20).mean() > preco.rolling(100).mean())
    r, _ = retorno_do_sinal(s.astype(float), r_a, ret["cdi"])
    bt = cruzamento(preco, ret["cdi"])
    assert np.allclose(r, bt["retorno"])


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("teste_"):
            f()
            print("ok", nome)
