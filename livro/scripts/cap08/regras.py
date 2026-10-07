"""Regras de negociação do Capítulo 8.

Cada função devolve a posição-alvo (fração do capital),
decidida com os dados disponíveis até o fechamento do dia.
"""
import numpy as np
import pandas as pd


def tendencia(p, n=200):
    """Comprado enquanto o preço supera a média de n dias."""
    return (p > p.rolling(n).mean()).astype(float)


def estado(entra, sai):
    """Posição 1 do sinal de entrada até o sinal de saída."""
    pos = pd.Series(np.nan, index=entra.index)
    pos[entra] = 1.0
    pos[sai] = 0.0
    return pos.ffill().fillna(0.0)


def zscore(p, n=20):
    """Desvio do log-preço à média móvel, em desvios."""
    x = np.log(p)
    return (x - x.rolling(n).mean()) / x.rolling(n).std()


def reversao(p, n=20, entrada=2.0):
    """Compra quando z < -entrada; sai quando z volta a zero."""
    z = zscore(p, n)
    return estado(z < -entrada, z >= 0)


def meia_vida(x):
    """Meia-vida (em dias) de um AR(1) ajustado à série x."""
    x = x.dropna()
    phi = np.polyfit(x[:-1], x[1:], 1)[0]
    return -np.log(2) / np.log(phi), phi


def donchian(ohlc, n=20):
    """Compra no rompimento da máxima de n dias; sai quando o
    fechamento perde a mínima dos últimos n/2 dias."""
    teto = ohlc["High"].rolling(n).max().shift(1)
    piso = ohlc["Low"].rolling(n // 2).min().shift(1)
    c = ohlc["Close"]
    return estado(c > teto, c < piso)
