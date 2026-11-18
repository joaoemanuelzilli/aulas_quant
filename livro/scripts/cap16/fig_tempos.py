"""Figura do Capítulo 16: tempo de execução por tamanho de entrada.

Lê dados/c16_tempos.csv (gravado por medir_tempos.py).
"""
import sys

sys.path.insert(0, "..")
from estilo import *  # noqa: E402,F403

t = pd.read_csv(DADOS / "c16_tempos.csv")

fig, (a1, a2) = figura(altura=66, ncols=2)

neg = t[t.experimento == "negocio"]
for metodo, cor, nome in (("ingenuo", OCRE, "todos os pares\nO(n²)"),
                          ("uma_passada", ROXO, "uma passada\nO(n)")):
    s = neg[neg.metodo == metodo]
    a1.loglog(s.n, s.segundos, "o-", color=cor, ms=2.6)
    rotulo_final(a1, s.n.iloc[-1], s.segundos.iloc[-1], nome, cor)
a1.set_title("Melhor compra e venda")
a1.set_xlabel("número de preços n")
a1.set_ylabel("segundos por chamada")
a1.set_xlim(80, 1e8)
a1.set_xticks([1e2, 1e3, 1e4, 1e5, 1e6])

lr = t[t.experimento == "logret"]
for metodo, cor, nome, dy in (("laco", OCRE, "laço em\nPython", 7),
                              ("numpy", ROXO, "NumPy", 0)):
    s = lr[lr.metodo == metodo]
    a2.loglog(s.n, s.segundos, "o-", color=cor, ms=2.6)
    a2.annotate(nome, (s.n.iloc[-1], s.segundos.iloc[-1]),
                xytext=(4, dy), textcoords="offset points",
                va="center", fontsize=7.2, color=TINTA)
a2.set_title("Log-retornos de n preços")
a2.set_xlabel("número de preços n")
a2.set_xlim(5, 2e9)
a2.set_xticks([1e1, 1e3, 1e5, 1e7])
a2.set_ylim(top=1)
a1.set_ylim(top=3)
for ax in (a1, a2):
    ax.grid(True, which="major", axis="both")
    ax.minorticks_off()

salvar(fig, "cap16/tempos")
