"""Funções do Capítulo 5: título prefixado, derivada numérica,
Black-Scholes e gregas, gradiente descendente."""
import numpy as np
from scipy.stats import norm


# ---------------------------------------------- título prefixado
def preco_ltn(taxa, du, face=1000.0):
    """Preço de um prefixado sem cupom: face/(1+taxa)^(du/252)."""
    return face / (1 + taxa) ** (du / 252)


def duration_mod(taxa, du):
    """Duration modificada: -P'(taxa)/P."""
    return (du / 252) / (1 + taxa)


def convexidade(taxa, du):
    """Convexidade: P''(taxa)/P."""
    n = du / 252
    return n * (n + 1) / (1 + taxa) ** 2


# ---------------------------------------------- derivada numérica
def derivada(f, x, h=1e-5, tipo="central"):
    """Derivada de f em x por diferenças finitas."""
    if tipo == "frente":
        return (f(x + h) - f(x)) / h
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------- Black-Scholes
def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) \
        / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs(S, K, T, r, sigma, tipo="compra"):
    """Preço de Black-Scholes de uma opção europeia."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    desc = K * np.exp(-r * T)
    if tipo == "compra":
        return S * norm.cdf(d1) - desc * norm.cdf(d2)
    return desc * norm.cdf(-d2) - S * norm.cdf(-d1)


def gregas(S, K, T, r, sigma):
    """Gregas da opção de compra europeia (theta por ano)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    dens, desc = norm.pdf(d1), K * np.exp(-r * T)
    return {
        "delta": norm.cdf(d1),
        "gama": dens / (S * sigma * np.sqrt(T)),
        "vega": S * np.sqrt(T) * dens,
        "theta": (-S * dens * sigma / (2 * np.sqrt(T))
                  - r * desc * norm.cdf(d2)),
        "rho": T * desc * norm.cdf(d2),
    }


# ---------------------------------------------- otimização
def gradiente_descendente(grad, x0, eta, passos):
    """Devolve a trajetória x_0, x_1, ..., x_passos."""
    x = np.asarray(x0, dtype=float)
    caminho = [x]
    for _ in range(passos):
        x = x - eta * grad(x)
        caminho.append(x)
    return np.array(caminho)
