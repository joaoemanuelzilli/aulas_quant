"""Aproximações de 1a e 2a ordem da variação de preço."""
import sys; sys.path.insert(0, "..")
from estilo import *
from calculo import convexidade, duration_mod, preco_ltn

y0, du = 0.1436, 2520
D, C = duration_mod(y0, du), convexidade(y0, du)
pb = np.linspace(-400, 400, 161)
dy = pb / 10_000
exata = preco_ltn(y0 + dy, du) / preco_ltn(y0, du) - 1
o1 = -D * dy
o2 = o1 + 0.5 * C * dy**2
fig, (a1, a2) = figura(altura=62, ncols=2)
a1.plot(pb, exata, color=ROXO, lw=2.2)
a1.plot(pb, o1, color=OCRE, lw=1.0, ls="--")
a1.plot(pb, o2, color=VERDE, lw=0.9, ls=":")
rotulo_final(a1, pb[-1], o1[-1], "1ª ordem", OCRE, dx=-38)
a1.annotate("exata e 2ª ordem", (pb[25], exata[25]), xytext=(6, 4),
            textcoords="offset points", fontsize=7.2)
a1.set_xlabel("choque na taxa (pb)")
a1.set_ylabel("variação do preço")
pct(a1, "y")
a1.set_title("Título de 10 anos", fontsize=7.6)
for du_, cor, nome in [(2520, ROXO, "10 anos"),
                       (532, OCRE, "532 du")]:
    D_, C_ = duration_mod(y0, du_), convexidade(y0, du_)
    ex = preco_ltn(y0 + dy, du_) / preco_ltn(y0, du_) - 1
    a2.plot(pb, -D_ * dy - ex, color=cor, lw=1.2)
    a2.plot(pb, -D_ * dy + 0.5 * C_ * dy**2 - ex, color=cor,
            lw=0.9, ls=":")
    k = 20 if du_ == 2520 else -12
    a2.annotate(nome, (pb[k], (-D_ * dy - ex)[k]),
                xytext=(6, -8) if du_ == 2520 else (-4, -9),
                textcoords="offset points", fontsize=7.2,
                ha="left" if du_ == 2520 else "right")
a2.axhline(0, color=CINZA, lw=0.5)
a2.set_xlabel("choque na taxa (pb)")
a2.set_ylabel("aproximação menos exata (p.p.)")
a2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: f"{v*100:.0f}".replace("-", "−")))
a2.set_title("Erro das aproximações", fontsize=7.6)
salvar(fig, "cap05/taylor")
