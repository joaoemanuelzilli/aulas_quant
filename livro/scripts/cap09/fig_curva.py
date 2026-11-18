"""Contratos 1 e 4 do WTI e o spread entre eles, 1990 a 2024."""
import sys; sys.path.insert(0, "..")
from estilo import *
from futuros import ler

f = ler(["cl1", "cl4"], "1990").dropna()
s = f.cl4 - f.cl1
fig, (a, b) = figura(altura=88, nrows=2, sharex=True,
                     gridspec_kw={"height_ratios": [1.15, 1]})
a.plot(f.index, f.cl1.clip(lower=0), color=ROXO, lw=0.8)
a.plot(f.index, f.cl4, color=OCRE, lw=0.8)
a.set_title("Preço (US$ por barril)")
a.set_ylim(0, 150)
a.text(pd.Timestamp("1991-06-01"), 128, "contrato 1", color=TINTA, fontsize=7.2)
a.plot([pd.Timestamp("1990-07-01"), pd.Timestamp("1991-03-01")], [131, 131], color=ROXO, lw=1.2)
a.text(pd.Timestamp("1991-06-01"), 112, "contrato 4", color=TINTA, fontsize=7.2)
a.plot([pd.Timestamp("1990-07-01"), pd.Timestamp("1991-03-01")], [115, 115], color=OCRE, lw=1.2)
a.annotate("20/04/2020: contrato 1 a $-$37,63", (pd.Timestamp("2020-04-20"), 0),
           xytext=(pd.Timestamp("2005-06-01"), 8), fontsize=6.8, color=CINZA,
           arrowprops=dict(arrowstyle="-", color=CINZA, lw=0.5))
LIM = 16
b.fill_between(s.index, 0, s.clip(0, LIM), color=ROXO, lw=0)
b.fill_between(s.index, 0, s.clip(-LIM, 0), color=OCRE, lw=0)
b.axhline(0, color=CINZA, lw=0.5)
b.set_ylim(-LIM, LIM)
b.set_title("Contrato 4 menos contrato 1 (US$ por barril)")
b.text(pd.Timestamp("1990-06-01"), 11, "contango", fontsize=7.2)
b.text(pd.Timestamp("1990-06-01"), -14, "backwardation", fontsize=7.2)
b.annotate("+66,14 em 20/04/2020", (pd.Timestamp("2020-04-20"), LIM),
           xytext=(pd.Timestamp("2006-01-01"), 13.2), fontsize=6.8, color=CINZA,
           arrowprops=dict(arrowstyle="-", color=CINZA, lw=0.5))
virgula(b, casas=0)
salvar(fig, "cap09/curva")
print("dias com spread fora da escala:", int((s.abs() > LIM).sum()),
      s[s.abs() > LIM].round(2).to_dict())
