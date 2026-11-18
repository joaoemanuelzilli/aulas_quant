"""Algoritmos do Capítulo 16: a versão ingênua e a rápida."""
import numpy as np


# --- melhor compra e venda -------------------------------------
def melhor_negocio_ingenuo(p):
    """Maior p[j] - p[i] com i < j, testando todos os pares."""
    melhor, compra, venda = 0.0, 0, 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[j] - p[i] > melhor:
                melhor, compra, venda = p[j] - p[i], i, j
    return melhor, compra, venda


def melhor_negocio(p):
    """O mesmo em uma passada: tempo O(n), memória O(1)."""
    melhor, compra, venda = 0.0, 0, 0
    i_min = 0                       # índice do menor preço até j
    for j in range(1, len(p)):
        if p[j] - p[i_min] > melhor:
            melhor, compra, venda = p[j] - p[i_min], i_min, j
        if p[j] < p[i_min]:
            i_min = j
    return melhor, compra, venda


# --- soma de dois ----------------------------------------------
def dois_somam_ingenuo(a, alvo):
    """Par i < j com a[i] + a[j] == alvo, testando todos."""
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] + a[j] == alvo:
                return i, j
    return None


def dois_somam(a, alvo):
    """O mesmo com um dicionário: tempo O(n) em média."""
    visto = {}                  # valor -> índice em que apareceu
    for j, x in enumerate(a):
        if alvo - x in visto:
            return visto[alvo - x], j
        visto[x] = j
    return None


# --- drawdown máximo -------------------------------------------
def drawdown_maximo_ingenuo(v):
    """Para cada t, recalcula o pico de v[0..t]: O(n^2)."""
    return min(v[t] / max(v[:t + 1]) - 1 for t in range(len(v)))


def drawdown_maximo(v):
    """O mesmo carregando o pico: tempo O(n), memória O(1)."""
    pico, pior = v[0], 0.0
    for x in v:
        pico = max(pico, x)
        pior = min(pior, x / pico - 1)
    return pior


# --- média móvel -----------------------------------------------
def media_movel_ingenua(x, k):
    """Soma k valores para cada data: O(n k)."""
    return [sum(x[t - k + 1:t + 1]) / k
            for t in range(k - 1, len(x))]


def media_movel(x, k):
    """Soma que desliza: entra um valor, sai outro. O(n)."""
    saida, soma = [], 0.0
    for t, xt in enumerate(x):
        soma += xt
        if t >= k:
            soma -= x[t - k]        # sai o valor mais antigo
        if t >= k - 1:
            saida.append(soma / k)
    return saida


def media_movel_np(x, k):
    """Soma acumulada: S[t+1] - S[t+1-k] é a soma da janela."""
    s = np.concatenate(([0.0], np.cumsum(x)))
    return (s[k:] - s[:-k]) / k


# --- busca binária ---------------------------------------------
def busca_linear(a, x):
    """Primeiro i com a[i] >= x, olhando um por um: O(n)."""
    for i, v in enumerate(a):
        if v >= x:
            return i
    return len(a)


def busca_binaria(a, x):
    """Primeiro i com a[i] >= x, com a ordenado: O(log n)."""
    lo, hi = 0, len(a)              # a resposta está em [lo, hi]
    while lo < hi:
        meio = (lo + hi) // 2
        if a[meio] < x:
            lo = meio + 1       # a resposta está à direita
        else:
            hi = meio           # meio ainda pode ser a resposta
    return lo
