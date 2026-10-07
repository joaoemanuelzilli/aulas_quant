"""QQ-plots do Ibovespa: contra a normal e contra a t ajustada."""
import sys; sys.path.insert(0, "..")
from estilo import *
from scipy import stats

r = np.log(precos("^BVSP")["^BVSP"].dropna()).diff().dropna()
n = len(r)
emp = np.sort(r.values)
prob = (np.arange(1, n + 1) - 0.5) / n
nu, loc, esc = stats.t.fit(r)
teo = {"Contra a normal": stats.norm.ppf(prob, r.mean(), r.std()),
       f"Contra a $t$ ajustada ($\\nu = {nu:.1f}$)".replace(".", ","): stats.t.ppf(prob, nu, loc, esc)}
fig, eixos = figura(altura=58, ncols=2, sharey=True)
for ax, (tit, q) in zip(eixos, teo.items()):
    ax.plot([-0.17, 0.17], [-0.17, 0.17], color=CINZA, lw=0.8)
    ax.plot(q, emp, "o", ms=1.9, color=ROXO, alpha=0.75, mew=0)
    ax.set_title(tit)
    ax.set_xlabel("quantil teórico")
    ax.set_xlim(-0.115, 0.115); ax.set_ylim(-0.175, 0.145)
    ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    pct(ax, "x"); pct(ax, "y")
    print(tit, "quantis teóricos extremos:", q[0], q[-1], "empíricos:", emp[0], emp[-1])
eixos[0].set_ylabel("quantil observado")
eixos[0].annotate("quedas maiores do\nque a normal admite", (-0.052, -0.125), xytext=(-0.02, -0.15),
                  fontsize=6.8, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
salvar(fig, "cap03/qq")
