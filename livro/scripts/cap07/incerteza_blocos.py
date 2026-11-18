"""Bootstrap em blocos: incerteza com dependência entre dias."""
import numpy as np
from carteira import perda, cota

rng = np.random.default_rng(11)


def blocos(x, tam, n=None):
    """Reamostra x em blocos de `tam` observações seguidas."""
    n = n or len(x)
    ini = rng.integers(0, len(x) - tam + 1, size=n // tam + 1)
    return np.concatenate([x[i:i + tam] for i in ini])[:n]


def medidas(x):
    v95, v99 = np.quantile(x, [0.95, 0.99])
    es99 = x[x >= v99].mean()
    return x.std(ddof=1), v95, v99, es99, es99 / v99


L = perda["2015":"2024"].values / 1000
nomes = ["desvio-padrão", "VaR 95%", "VaR 99%", "ES 99%", "ES/VaR 99%"]
for rot, tam in (("i.i.d.", 1), ("blocos de 21", 21)):
    b = np.array([medidas(blocos(L, tam)) for _ in range(5000)])
    lo, hi = np.percentile(b, [5, 95], axis=0)
    print(rot, " | ".join(f"{n}: {a:.1f} a {c:.1f}" if n != "ES/VaR 99%" else f"{n}: {a:.2f} a {c:.2f}"
                          for n, a, c in zip(nomes, lo, hi)))

# drawdown máximo e Calmar: blocos de 63 pregões dos retornos da cota
r = cota.pct_change().dropna().values
anos = len(r) / 252
res = []
for _ in range(5000):
    c = np.cumprod(1 + blocos(r, 63))
    mdd = (c / np.maximum.accumulate(c) - 1).min()
    res.append((mdd, (c[-1] ** (1 / anos) - 1) / abs(mdd)))
res = np.array(res)
print("drawdown máximo, IC 90%:", np.percentile(res[:, 0], [5, 95]).round(3),
      " Calmar, IC 90%:", np.percentile(res[:, 1], [5, 95]).round(2))

# passeio aleatório normal com a mesma média e o mesmo desvio diários
fr = []
for _ in range(1000):
    c = np.cumprod(1 + rng.normal(r.mean(), r.std(), len(r)))
    fr.append((c < np.maximum.accumulate(c)).mean())
print("fração abaixo do pico em passeio aleatório: média", round(np.mean(fr), 3),
      " faixa de 90%:", np.percentile(fr, [5, 95]).round(3))
