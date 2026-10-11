"""Linha de alocação de capital com os ativos A e B."""
import sys; sys.path.insert(0, "..")
from estilo import *

mu = np.array([0.10, 0.15]); dp = np.array([0.20, 0.30])
rho, rf = 0.3, 0.05
cov = np.array([[dp[0]**2, rho * dp[0] * dp[1]],
                [rho * dp[0] * dp[1], dp[1]**2]])
w = np.linspace(0, 1, 201)
W = np.column_stack([w, 1 - w])
m, s = W @ mu, np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
tg = np.linalg.solve(cov, mu - rf); tg /= tg.sum()
pontos = {"A": (dp[0], mu[0]), "B": (dp[1], mu[1]),
          "tangente": (np.sqrt(tg @ cov @ tg), tg @ mu)}
fig, ax = figura(altura=66)
ax.plot(s, m, color=ROXO, lw=1.4, zorder=3)
x = np.array([0, 0.345])
for (nome, (sx, mx)), cor in zip(pontos.items(), [CINZA, VERDE, OCRE]):
    incl = (mx - rf) / sx
    ax.plot(x, rf + incl * x, color=cor, lw=1.0, zorder=2)
    ax.plot(sx, mx, "o", color=TINTA, ms=3.8, zorder=5)
    ax.annotate("Sharpe " + f"{incl:.2f}".replace(".", ","),
                (x[1], rf + incl * x[1]), xytext=(4, 0), va="center",
                textcoords="offset points", fontsize=7)
    print(nome, "vol", round(sx, 4), "retorno", round(mx, 4),
          "Sharpe", round(incl, 3))
print("pesos da tangente (A, B):", tg.round(3))
print("50/50: Sharpe", round((m[100] - rf) / s[100], 3))
for alvo, (nome, (sx, mx)) in zip([0.10] * 3, pontos.items()):
    a = alvo / sx
    print(f"10% de vol com {nome}: fração {a:.3f}, retorno "
          f"{rf + a * (mx - rf):.4f}")
ax.plot(0, rf, "o", color=TINTA, ms=4, clip_on=False, zorder=6)
nota = dict(textcoords="offset points", fontsize=7)
ax.annotate("ativo livre\nde risco (5%)", (0, rf), xytext=(7, -4),
            va="top", **nota)
ax.annotate("A", (dp[0], mu[0]), xytext=(5, -8), **nota)
ax.annotate("B", (dp[1], mu[1]), xytext=(4, -9), **nota)
ax.annotate("carteira tangente\n(47% em A, 53% em B)", pontos["tangente"],
            xytext=(-62, 20), **nota,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.set_xlim(0, 0.41); ax.set_ylim(0.025, 0.185)
ax.spines["bottom"].set_bounds(0, 0.35)
ax.set_xticks([0, 0.1, 0.2, 0.3])
ax.set_xlabel("volatilidade"); ax.set_ylabel("retorno esperado")
pct(ax, "x"); pct(ax, "y")
salvar(fig, "cap06/alocacao")
