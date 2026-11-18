"""Curvas de nível da variância de dois ativos e o gradiente."""
import sys; sys.path.insert(0, "..")
from estilo import *

s_a, s_b, rho = 0.20, 0.30, 0.30
S = np.array([[s_a**2, rho * s_a * s_b], [rho * s_a * s_b, s_b**2]])
g1, g2 = np.meshgrid(np.linspace(-1, 1.5, 251),
                     np.linspace(-1, 1.5, 251))
V = S[0, 0] * g1**2 + 2 * S[0, 1] * g1 * g2 + S[1, 1] * g2**2
fig, ax = figura(altura=70)
niveis = [0.005, 0.02, 0.04, 0.07, 0.11, 0.16, 0.22]
cs = ax.contour(g1, g2, V, levels=niveis, colors=CINZA,
                linewidths=0.6)
ax.clabel(cs, fmt=lambda v: f"{v:.3f}".replace(".", ","),
          fontsize=6.2, inline_spacing=2)
pts = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-0.6, 0.8],
                [0.5, -0.7], [1.2, -0.5]])
grad = 2 * pts @ S
esc = 2.2
ax.quiver(pts[:, 0], pts[:, 1], grad[:, 0], grad[:, 1],
          angles="xy", scale_units="xy", scale=1 / esc, color=ROXO,
          width=0.006, headwidth=4, headlength=5)
ax.plot(pts[:, 0], pts[:, 1], "o", color=TINTA, ms=3)
ax.plot(0, 0, "o", color=OCRE, ms=4)
ax.annotate("mínimo: $w_1 = w_2 = 0$", (0, 0), xytext=(6, -10),
            textcoords="offset points", fontsize=7.2)
ax.set_aspect("equal")
ax.set_xlim(-1, 1.5); ax.set_ylim(-1, 1.5)
ax.grid(False)
ax.set_xlabel("peso no ativo A, $w_1$")
ax.set_ylabel("peso no ativo B, $w_2$")
virgula(ax, "x"); virgula(ax, "y")
salvar(fig, "cap05/curvas_nivel")
