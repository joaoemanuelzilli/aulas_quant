"""Problemas de probabilidade do Capítulo 16: conta e simulação."""
import math

import numpy as np

SIMS = 1_000_000


# 1. dado com direito a uma segunda jogada
def dado_segunda_chance():
    # fica com a 1a jogada se ela for maior que E[2a] = 3,5
    return 0.5 * 3.5 + 0.5 * (4 + 5 + 6) / 3


def dado_tres_jogadas():
    e2 = dado_segunda_chance()          # valor de ter 2 jogadas
    fica = [x for x in range(1, 7) if x > e2]
    return (sum(fica) + (6 - len(fica)) * e2) / 6


def dado_simulado(semente=2026):
    rng = np.random.default_rng(semente)
    d = rng.integers(1, 7, size=(SIMS, 2))
    ganho = np.where(d[:, 0] >= 4, d[:, 0], d[:, 1])
    return ganho.mean(), ganho.std() / math.sqrt(SIMS)


def dado_sementes(n_sementes=400):
    """Distância ao valor exato, em erros-padrão, com várias sementes.

    A semente 16, a primeira usada, deu 4,2548 (3,1 erros-padrão);
    esta função mostra que isso é acaso, e não defeito do código.
    """
    z = []
    for s in range(n_sementes):
        m, ep = dado_simulado(s)
        z.append((m - 4.25) / ep)
    return np.array(z)


# 2. espera por duas caras seguidas (CC) e por cara-coroa (CK)
def espera_simulada(padrao, sims=200_000, semente=2026):
    rng = np.random.default_rng(semente)
    # 200 lançamentos bastam: P(espera > 200) é desprezível
    m = rng.integers(0, 2, size=(sims, 200))   # 1 = cara
    alvo = [1 if c == "C" else 0 for c in padrao]
    ok = (m[:, :-1] == alvo[0]) & (m[:, 1:] == alvo[1])
    assert ok.any(axis=1).all()
    t = ok.argmax(axis=1) + 2          # lançamento em que termina
    return t.mean(), t.std() / math.sqrt(sims)


# 3. Bayes: uma moeda de duas caras entre mil
def bayes_moeda(n_moedas=1000, caras=10):
    a_priori = 1 / n_moedas
    vero_viciada, vero_honesta = 1.0, 0.5 ** caras
    num = a_priori * vero_viciada
    return num / (num + (1 - a_priori) * vero_honesta)


def bayes_moeda_simulada(n_moedas=1000, caras=10, sims=2_000_000,
                         semente=2026):
    rng = np.random.default_rng(semente)
    viciada = rng.random(sims) < 1 / n_moedas
    k = np.where(viciada, caras, rng.binomial(caras, 0.5, sims))
    so_caras = k == caras
    p = viciada[so_caras].mean()
    return p, math.sqrt(p * (1 - p) / so_caras.sum()), so_caras.sum()


# 4. ruína do jogador: passeio simétrico entre -b e +a
def ruina(a, b):
    return b / (a + b), a * b       # P(chegar a +a), passos médios


def ruina_simulada(a, b, sims=200_000, semente=2026):
    rng = np.random.default_rng(semente)
    pos = np.zeros(sims, dtype=int)
    passos = np.zeros(sims, dtype=int)
    vivo = np.ones(sims, dtype=bool)
    while vivo.any():
        k = vivo.sum()
        pos[vivo] += rng.choice((-1, 1), size=k)
        passos[vivo] += 1
        vivo &= (pos < a) & (pos > -b)
    return (pos == a).mean(), passos.mean()


# 5. o maior de n sorteios uniformes
def maximo_uniformes(n):
    # P(M <= x) = x^n; E[M] = integral de P(M > x) = 1 - 1/(n+1)
    return n / (n + 1)


def maximo_uniformes_simulado(n, sims=200_000, semente=2026):
    rng = np.random.default_rng(semente)
    m = rng.random((sims, n)).max(axis=1)
    return m.mean(), m.std() / math.sqrt(sims)


if __name__ == "__main__":
    print(f"dado, 2 jogadas: {dado_segunda_chance():.4f}"
          f"  3 jogadas: {dado_tres_jogadas():.4f}")
    m, ep = dado_simulado()
    print(f"  simulado: {m:.4f} (erro-padrão {ep:.4f})")
    m16, ep16 = dado_simulado(16)
    print(f"  semente 16: {m16:.4f} ({(m16 - 4.25) / ep16:.1f} ep)")
    z = np.abs(dado_sementes())
    print(f"  400 sementes: {np.mean(z > 2):.1%} além de 2 ep, "
          f"{np.sum(z > 3)} além de 3 ep")
    for padrao in ("CC", "CK"):
        m, ep = espera_simulada(padrao)
        print(f"espera {padrao}: {m:.3f} (erro-padrão {ep:.3f})")
    print(f"Bayes, moeda viciada: {bayes_moeda():.4f}",
          f"próxima cara: {bayes_moeda() + 0.5 * (1 - bayes_moeda()):.4f}")
    p, ep, casos = bayes_moeda_simulada()
    print(f"  simulado: {p:.4f} (erro-padrão {ep:.4f}) em {casos}")
    for a, b in ((4, 2), (10, 10)):
        p, t = ruina(a, b)
        ps, ts = ruina_simulada(a, b)
        print(f"ruína a={a} b={b}: P={p:.4f} T={t}"
              f"  simulado P={ps:.4f} T={ts:.2f}")
    for n in (1, 10, 100):
        m, ep = maximo_uniformes_simulado(n)
        print(f"máximo de {n} uniformes: {maximo_uniformes(n):.4f}"
              f"  simulado {m:.4f} (erro-padrão {ep:.4f})")
