"""Problemas de cálculo e de álgebra linear do Capítulo 16."""
import math

import numpy as np
import sympy as sp
from scipy import integrate, stats


# cálculo 1: E[max(Z, 0)] e a call no dinheiro
def media_parte_positiva():
    z = sp.symbols("z", real=True)
    dens = sp.exp(-z ** 2 / 2) / sp.sqrt(2 * sp.pi)
    return sp.simplify(sp.integrate(z * dens, (z, 0, sp.oo)))


def call_bs_sem_juros(s, sigma, t):
    """Black-Scholes com r = 0 e K = S: S [N(d1) - N(d2)]."""
    d1 = sigma * math.sqrt(t) / 2
    return s * (stats.norm.cdf(d1) - stats.norm.cdf(-d1))


def call_aproximada(s, sigma, t):
    return s * sigma * math.sqrt(t) / math.sqrt(2 * math.pi)


# cálculo 2: a regra dos 72
def dobra_exato(r):
    return math.log(2) / math.log(1 + r)


def dobra_taylor(r):
    # ln(1+r) ~ r(1 - r/2)  e  1/(1-u) ~ 1 + u
    return math.log(2) / r * (1 + r / 2)


# álgebra 1: faixa possível de uma correlação
def faixa_correlacao(r12, r23):
    folga = math.sqrt((1 - r12 ** 2) * (1 - r23 ** 2))
    return r12 * r23 - folga, r12 * r23 + folga


def faixa_correlacao_numerica(r12, r23, passo=1e-4):
    validos = []
    for r13 in np.arange(-1, 1 + passo / 2, passo):
        c = np.array([[1, r12, r13], [r12, 1, r23], [r13, r23, 1]])
        if np.linalg.eigvalsh(c).min() >= -1e-12:
            validos.append(r13)
    return min(validos), max(validos)


def determinante_simbolico():
    a, b, c = sp.symbols("rho12 rho13 rho23", real=True)
    m = sp.Matrix([[1, a, b], [a, 1, c], [b, c, 1]])
    return sp.expand(m.det())


# álgebra 2: autovalores da matriz de correlação constante
def equicorrelacao(n, rho):
    c = (1 - rho) * np.eye(n) + rho * np.ones((n, n))
    return np.linalg.eigvalsh(c)


def variancia_pesos_iguais(n, rho, sigma=1.0):
    c = sigma ** 2 * ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))
    w = np.full(n, 1 / n)
    return w @ c @ w, sigma ** 2 * (1 + (n - 1) * rho) / n


if __name__ == "__main__":
    print("E[max(Z,0)] =", media_parte_positiva(),
          f"= {float(media_parte_positiva()):.4f}")
    for t in (1.0, 0.25):
        print(f"call S=100 sigma=30% T={t}: "
              f"BS {call_bs_sem_juros(100, 0.3, t):.3f}  "
              f"aprox {call_aproximada(100, 0.3, t):.3f}")
    for r in (0.08, 0.12):
        print(f"dobra a {r:.0%}: exato {dobra_exato(r):.3f}  "
              f"Taylor {dobra_taylor(r):.3f}  72/R {72 / (100 * r):.3f}"
              f"  69,3/R {0.693 / r:.3f}")
    print(f"69,3 x (1 + 0,04) = {69.3 * 1.04:.1f}")
    print("det:", determinante_simbolico())
    print("faixa de rho13 (0,9 e 0,9):", faixa_correlacao(0.9, 0.9),
          faixa_correlacao_numerica(0.9, 0.9))
    print("faixa de rho13 (0,5 e 0,5):", faixa_correlacao(0.5, 0.5),
          faixa_correlacao_numerica(0.5, 0.5))
    print("autovalores n=5 rho=0,3:", equicorrelacao(5, 0.3).round(6))
    print("menor rho n=5:", -1 / 4, equicorrelacao(5, -0.25).min(),
          equicorrelacao(5, -0.26).min())
    print("variância pesos iguais n=5 rho=0,3:",
          variancia_pesos_iguais(5, 0.3))
