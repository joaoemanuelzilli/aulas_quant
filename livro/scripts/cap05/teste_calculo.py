"""Testes das funções de calculo.py.  Rode: python teste_calculo.py"""
import numpy as np

from calculo import (bs, convexidade, derivada, duration_mod,
                     gradiente_descendente, gregas, preco_ltn)

# 1. LTN: exemplo da metodologia do Tesouro e vencimento
assert np.floor(preco_ltn(0.1436, 532) * 1e6) / 1e6 == 753.315323
assert preco_ltn(0.2, 0) == 1000.0
# um ano de 252 dias úteis a 10%: 1000/1,1
assert np.isclose(preco_ltn(0.10, 252), 1000 / 1.1)

# 2. duration e convexidade contra diferenças finitas
for taxa, du in [(0.05, 100), (0.1436, 532), (0.12, 2520)]:
    f = lambda y: preco_ltn(y, du)
    P = f(taxa)
    assert np.isclose(-derivada(f, taxa) / P,
                      duration_mod(taxa, du), rtol=1e-7)
    h = 1e-4
    seg = (f(taxa + h) - 2 * P + f(taxa - h)) / h**2
    assert np.isclose(seg / P, convexidade(taxa, du), rtol=1e-5)

# 3. derivada numérica: ordens de erro em e^x no zero
for h in (1e-2, 1e-3):
    erro_fr = abs(derivada(np.exp, 0.0, h, "frente") - 1)
    erro_ce = abs(derivada(np.exp, 0.0, h) - 1)
    assert np.isclose(erro_fr, h / 2, rtol=0.01)       # ~ h f''/2
    assert np.isclose(erro_ce, h**2 / 6, rtol=0.01)    # ~ h^2 f'''/6

# 4. Black-Scholes: paridade, limites e gregas contra diferenças
rng = np.random.default_rng(0)
for _ in range(200):
    S, K = rng.uniform(5, 100, 2)
    T, r, s = rng.uniform(0.02, 3), rng.uniform(0, 0.2), \
        rng.uniform(0.05, 0.8)
    C, P = bs(S, K, T, r, s), bs(S, K, T, r, s, "venda")
    assert np.isclose(C - P, S - K * np.exp(-r * T))
    assert max(S - K * np.exp(-r * T), 0) - 1e-9 <= C <= S
    g = gregas(S, K, T, r, s)
    e = 1e-4
    num = {
        "delta": derivada(lambda x: bs(x, K, T, r, s), S, e * S),
        "vega": derivada(lambda x: bs(S, K, T, r, x), s, e),
        "rho": derivada(lambda x: bs(S, K, T, x, s), r, e),
        "theta": -derivada(lambda x: bs(S, K, x, r, s), T, e),
    }
    for k, v in num.items():
        assert np.isclose(g[k], v, rtol=1e-4, atol=1e-6), k
    gam = (bs(S * (1 + e), K, T, r, s) - 2 * C
           + bs(S * (1 - e), K, T, r, s)) / (e * S) ** 2
    assert np.isclose(g["gama"], gam, rtol=1e-3, atol=1e-6)
# opção muito dentro do dinheiro: C ~ S - K e^(-rT), delta ~ 1
assert np.isclose(bs(100, 10, 1, 0.1, 0.2), 100 - 10 * np.exp(-0.1))
assert np.isclose(gregas(100, 10, 1, 0.1, 0.2)["delta"], 1)

# 5. gradiente descendente numa quadrática: converge para o mínimo
A = np.array([[3.0, 1.0], [1.0, 2.0]]); b = np.array([1.0, -1.0])
grad = lambda x: A @ x - b                  # f = x'Ax/2 - b'x
cam = gradiente_descendente(grad, [5, 5], 0.3, 200)
assert np.allclose(cam[-1], np.linalg.solve(A, b))
assert cam.shape == (201, 2)
# acima de 2/maior autovalor, diverge
lmax = np.linalg.eigvalsh(A).max()
cam = gradiente_descendente(grad, [5, 5], 2.1 / lmax, 200)
assert np.linalg.norm(cam[-1]) > 1e3
print("todos os testes de calculo.py passaram")
