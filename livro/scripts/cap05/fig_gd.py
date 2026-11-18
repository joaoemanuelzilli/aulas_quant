"""Caminho do gradiente descendente para três taxas."""
import runpy
import sys; sys.path.insert(0, "..")
from estilo import *
from calculo import gradiente_descendente

ns = runpy.run_path("l03_gradiente.py")
x, y, grad = ns["x"], ns["y"], ns["grad"]
a_, b_ = ns["a_mqo"], ns["b_mqo"]
A, B = np.meshgrid(np.linspace(-0.9, 0.9, 241),
                   np.linspace(-0.1, 1.22, 241))
L = ((y[None, None, :] - A[..., None]
      - B[..., None] * x[None, None, :]) ** 2).mean(axis=2)
fig, ax = figura(altura=74)
Lmin = L.min()
ax.contour(A, B, L, levels=Lmin + np.array([0.02, 0.1, 0.25, 0.5,
           0.8, 1.2]), colors=CINZA_CLARO, linewidths=0.6)
casos = [(0.05, OCRE, "$\\eta = 0{,}05$"), (0.5, VERDE,
         "$\\eta = 0{,}5$"), (0.95, ROXO, "$\\eta = 0{,}95$")]
for eta, cor, rot in casos:
    c = gradiente_descendente(grad, [0.6, 0], eta, 40)
    ax.plot(c[:, 0], c[:, 1], "-o", color=cor, lw=0.8, ms=2.2)
ax.plot(0.6, 0, "s", color=TINTA, ms=3.5)
ax.annotate("início (0,6; 0)", (0.6, 0), xytext=(-6, -3),
            textcoords="offset points", fontsize=7.2, ha="right")
ax.plot(a_, b_, "*", color=TINTA, ms=7, zorder=5)
ax.annotate("mínimo", (a_, b_), xytext=(-2, 9),
            textcoords="offset points", fontsize=7.2, ha="center")
ax.annotate("$\\eta = 0{,}05$: 40 passos\nnão bastam", (0.30, 0.27),
            xytext=(0.42, 0.42), fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.annotate("$\\eta = 0{,}5$", (-0.08, 0.6), xytext=(-0.55, 0.45),
            fontsize=7.2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.annotate("$\\eta = 0{,}95$: oscila em $a$", (-0.62, 0.95),
            xytext=(-0.85, 1.13), fontsize=7.2)
ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.1, 1.22)
ax.grid(False)
ax.set_xlabel("intercepto $a$ (p.p.)")
ax.set_ylabel("inclinação $b$")
virgula(ax, "x"); virgula(ax, "y")
salvar(fig, "cap05/gradiente_descendente")
