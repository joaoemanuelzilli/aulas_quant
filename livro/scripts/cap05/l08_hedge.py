import numpy as np

from calculo import bs, gregas

S0, K, r, sigma, mu = 30.0, 30.0, 0.14, 0.30, 0.20
n, dt, c = 63, 1 / 252, 0.001          # pregões, passo, custo
rng = np.random.default_rng(7)
z = rng.standard_normal((20_000, n))
lnS = np.log(S0) + np.cumsum((mu - sigma**2 / 2) * dt
                             + sigma * np.sqrt(dt) * z, axis=1)
S = np.hstack([np.full((len(z), 1), S0), np.exp(lnS)])
C0 = bs(S0, K, n * dt, r, sigma)


def vende_e_protege(a_cada):
    """Vende a opção e refaz o delta a cada `a_cada` pregões."""
    caixa, acoes, custo = np.full(len(S), C0), 0.0, 0.0
    for t in range(n):
        if a_cada and t % a_cada == 0:
            tau = (n - t) * dt              # prazo restante
            novo = gregas(S[:, t], K, tau, r, sigma)["delta"]
            caixa -= (novo - acoes) * S[:, t]
            custo += c * np.abs(novo - acoes) * S[:, t]
            acoes = novo
        caixa = caixa * np.exp(r * dt)
    custo += c * acoes * S[:, n]                 # zera as ações
    pl = caixa + acoes * S[:, n] - np.maximum(S[:, n] - K, 0)
    return pl / C0, (pl - custo) / C0


print(f"prêmio recebido: R$ {C0:.4f}")
print("refaz o delta     média   desvio  média com custo")
for nome, k in [("nunca", 0), ("a cada 5 dias", 5),
                ("todo dia", 1)]:
    pl, liq = vende_e_protege(k)
    print(f"{nome:<14} {pl.mean():+7.2%} {pl.std():7.2%}"
          f" {liq.mean():+9.2%}")
