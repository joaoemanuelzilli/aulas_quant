"""Monte Carlo: a estimativa converge e o erro cai como 1/raiz(N)."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy.stats import norm

mu, sigma = 0.10, 0.48
exato = norm.cdf((np.log(0.70) - mu) / sigma)
Ns = np.unique(np.logspace(1, 6, 41).astype(int))
rng = np.random.default_rng(seed=7)
est = np.array([np.mean(rng.normal(mu, sigma, N) < np.log(0.70)) for N in Ns])
ep = np.sqrt(exato * (1 - exato) / Ns)

fig, (a1, a2) = figura(altura=55, ncols=2)
g = np.logspace(1, 6, 300)
eg = np.sqrt(exato * (1 - exato) / g)
a1.fill_between(g, exato - 2 * eg, exato + 2 * eg, color=LILAS, lw=0)
a1.axhline(exato, color=CINZA, lw=0.6, ls="--")
a1.plot(Ns, est, "o", color=ROXO, ms=2.6)
a1.set_xscale("log")
a1.set_ylim(0, 0.4)
pct(a1)
a1.set_title("Estimativa de P(queda > 30%)")
a1.text(60, 0.36, "faixa: valor exato ± 2 erros-padrão", fontsize=6.8)
a2.loglog(Ns, np.abs(est - exato), "o", color=ROXO, ms=2.6)
a2.loglog(g, eg, color=TINTA, lw=0.8)
a2.annotate("erro-padrão\n√(p(1 − p)/N)", (3e4, np.sqrt(exato * (1 - exato) / 3e4)), xytext=(8, 14),
            textcoords="offset points", fontsize=6.8, arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a2.set_title("Erro absoluto (escala log)")
a2.set_ylim(1e-6, 0.3)
a2.set_yticks([1e-5, 1e-3, 1e-1]); a2.set_yticklabels(["0,00001", "0,001", "0,1"])
for a in (a1, a2):
    a.set_xlabel("número de cenários N")
    a.set_xticks([10, 1e3, 1e5]); a.set_xticklabels(["10", "1.000", "100.000"])
    a.minorticks_off()
salvar(fig, "cap03/mc")
print("exato", exato, "fração dentro de 2 ep:", np.mean(np.abs(est - exato) <= 2 * ep), len(Ns))
