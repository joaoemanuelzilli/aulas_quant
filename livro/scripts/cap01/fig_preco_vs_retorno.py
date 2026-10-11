"""Preço (não estacionário) contra retorno diário (oscila em torno de zero)."""
import sys; sys.path.insert(0, "..")
from estilo import *

# o retorno do primeiro pregão de 2015 precisa do último fechamento de 2014
p = precos("PETR4.SA", inicio="2014", fim="2024")["PETR4.SA"].dropna()
r = p.pct_change().dropna().loc["2015":]
p = p.loc["2015":]

fig, (a1, a2) = figura(altura=78, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
a1.plot(p.index, p, color=ROXO)
a1.set_title("Preço ajustado de PETR4 (R$)")
a2.plot(r.index, r, color=ROXO, lw=0.45)
a2.axhline(0, color=TINTA, lw=0.6)
a2.set_title("Retorno simples diário")
pct(a2)
pior = r.idxmin()
a2.annotate(f"{r.min():.1%}".replace(".", ",") + f"\n{pior:%d/%m/%Y}", (pior, r.min()), xytext=(12, 2),
            textcoords="offset points", fontsize=6.8, color=TINTA, va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
salvar(fig, "cap01/preco_vs_retorno")
print(f"pior dia: {pior:%d/%m/%Y} {r.min():.1%}; {len(r)} retornos; desvio-padrão diário {r.std():.2%}")
