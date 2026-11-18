"""Gás natural: spread de calendário e volatilidade por mês; efeito Samuelson."""
import sys; sys.path.insert(0, "..")
from estilo import *
from l06_sazonal import MESES, spread_por_mes, vol_por_mes, vol_por_prazo

sp, vol = spread_por_mes(), vol_por_mes()
fig, (a, b, c) = figura(altura=52, largura="larga", ncols=3,
                        gridspec_kw={"width_ratios": [1.25, 1.25, 0.8]})
x = np.arange(12)
a.bar(x, sp, color=[ROXO if v > 0 else OCRE for v in sp], width=0.7)
a.axhline(0, color=CINZA, lw=0.5)
a.set_title("Gás: $\\ln(F_2/F_1)$ médio, por mês")
pct(a, casas=0)
b.bar(x, vol, color=ROXO, width=0.7)
b.set_title("Henry Hub à vista: volatilidade")
pct(b, casas=0)
for ax in (a, b):
    ax.set_xticks(x); ax.set_xticklabels([m[0].upper() for m in MESES])
for fut, cor, nome in [("cl", ROXO, "WTI"), ("ng", OCRE, "gás")]:
    v = vol_por_prazo(fut)
    c.plot(v.index, v, "-o", color=cor, ms=3.5)
    rotulo_final(c, 3, v.iloc[-1], nome, cor, dx=6)
c.set_xticks([1, 2, 3]); c.set_xlim(0.7, 3.7); c.set_ylim(0.3, 0.5)
c.set_xlabel("contrato"); c.set_title("Volatilidade por prazo")
pct(c, casas=0)
salvar(fig, "cap09/sazonal")
