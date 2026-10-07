"""Previsão da volatilidade a partir de um dia de crise e de um dia calmo."""
import sys; sys.path.insert(0, "..")
from estilo import *
from arch import arch_model

r = 100 * np.log(precos("^BVSP")["^BVSP"].dropna()).diff().dropna()
res = arch_model(r, vol="GARCH", p=1, q=1).fit(disp="off")
w, al, be = res.params[["omega", "alpha[1]", "beta[1]"]]
vi = w / (1 - al - be)
s2 = res.conditional_volatility ** 2
H = 250
h = np.arange(1, H + 1)
origens = {"crise": s2.idxmax(), "calmaria": s2.idxmin()}
fig, ax = figura(altura=55)
lp = np.sqrt(252 * vi)
ax.axhline(lp, color=CINZA, lw=0.8, ls=(0, (4, 3)))
for (nome, d), c in zip(origens.items(), (ROXO, OCRE)):
    prev = vi + (al + be) ** (h - 1) * (s2[d] - vi)       # variância prevista para o dia h
    v = np.sqrt(252 * prev)
    ax.plot(h, v, color=c)
    mv = np.log(0.5) / np.log(al + be)
    print(f"{nome}: origem {d:%d/%m/%Y}  vol h=1 {v[0]:.1f}%  h=21 {v[20]:.1f}%  h=63 {v[62]:.1f}%  h=126 {v[125]:.1f}%  h=250 {v[-1]:.1f}%")
    ax.text(12 if nome == "crise" else 30, v[0] - (6 if nome == "crise" else 5), f"a partir de {d:%d/%m/%Y}", fontsize=7, va="center")
v = np.sqrt(252 * (vi + (al + be) ** (h - 1) * (s2.max() - vi)))
print("pico: distância da volatilidade ao longo prazo cai à metade em h =", int(np.argmax(v - lp <= (v[0] - lp) / 2)) + 1)
print(f"longo prazo {lp:.1f}%  meia-vida (variância) {mv:.1f} dias")
ax.text(248, lp - 11, f"longo prazo: {lp:.1f}%".replace(".", ","), fontsize=7, color=CINZA, ha="right")
ax.set_xlabel("horizonte $h$ (dias úteis à frente)"); ax.set_title("Volatilidade prevista para o dia $h$, anualizada (%)")
ax.set_xlim(0, H); ax.set_ylim(0, 130)
salvar(fig, "cap04/prev_vol")
