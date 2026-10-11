"""Fechamento sem ajuste contra fechamento ajustado por proventos."""
import sys; sys.path.insert(0, "..")
from estilo import *

# convenção do capítulo: o período "de 2019 a 2024" começa no último
# fechamento de 2018, de modo que o primeiro pregão de 2019 entra na conta
def serie(ajustados):
    s = precos("PETR4.SA", inicio="2018", fim="2024", ajustados=ajustados)["PETR4.SA"].dropna()
    return s.loc[s.loc["2018"].index[-1]:]

a, b = serie(True), serie(False)
fig, ax = figura(altura=60)
ax.plot(b.index, b, color=CINZA, lw=1)
ax.plot(a.index, a, color=ROXO)
ax.set_title("PETR4: o mesmo ativo, duas séries (R$)")
ax.text(pd.Timestamp("2019-02-01"), 37.5, "fechamento sem ajuste", fontsize=7.2, color=CINZA)
ax.text(pd.Timestamp("2022-03-01"), 5.5, "fechamento ajustado\npor proventos", fontsize=7.2)
salvar(fig, "cap01/ajuste_proventos")
ra, rb = a.iloc[-1] / a.iloc[0] - 1, b.iloc[-1] / b.iloc[0] - 1
print(f"{a.index[0]:%Y-%m-%d} a {a.index[-1]:%Y-%m-%d}: ajustado {ra:.1%}  sem ajuste {rb:.1%}; "
      f"sem ajuste ini/fim {b.iloc[0]:.2f} {b.iloc[-1]:.2f}; aj ini/fim {a.iloc[0]:.2f} {a.iloc[-1]:.2f}; "
      f"fator ajustado {1 + ra:.2f}")
