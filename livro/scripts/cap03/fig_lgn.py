"""Lei dos grandes números: a frequência de dias de alta de PETR4."""
import sys; sys.path.insert(0, "..")
from estilo import *

petr = precos("PETR4.SA")["PETR4.SA"].dropna()
alta = (petr.pct_change().dropna() > 0).to_numpy()
n = np.arange(1, len(alta) + 1)
freq = np.cumsum(alta) / n
p = alta.mean()
ep = np.sqrt(p * (1 - p) / n)

fig, ax = figura(altura=52)
ax.fill_between(n, p - 2 * ep, p + 2 * ep, color=LILAS, lw=0)
ax.plot(n, freq, color=ROXO, lw=1.0)
ax.axhline(p, color=CINZA, lw=0.6, ls="--")
ax.set_xscale("log")
ax.set_xlim(10, len(alta))
ax.set_ylim(0.15, 0.85)
ax.set_xticks([10, 100, 1000, len(alta)])
ax.set_xticklabels(["10", "100", "1.000", f"{len(alta):,}".replace(",", ".")])
ax.minorticks_off()
pct(ax)
ax.set_xlabel("pregões desde janeiro de 2010 (escala log)")
ax.set_ylabel("fração de dias de alta")
ax.text(14, 0.78, "faixa lilás: p ± 2√(p(1 − p)/n), dois erros-padrão", fontsize=6.8)
rotulo_final(ax, len(alta), p, f"{p * 100:.1f}%".replace(".", ","), CINZA, dx=3)
salvar(fig, "cap03/lgn")
for m in (20, 100, 1000, len(alta)):
    print(m, round(freq[m - 1], 4), "faixa ±", round(2 * ep[m - 1], 4))
print("fora da faixa (n>=100):", np.mean(np.abs(freq[99:] - p) > 2 * ep[99:]))
