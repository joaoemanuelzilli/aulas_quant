import matplotlib.pyplot as plt
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
petr = precos.loc["2024", "PETR4.SA"].dropna()
retorno = petr.pct_change().dropna()

# uma Figure com dois Axes empilhados, que dividem o eixo x
fig, (ax_preco, ax_ret) = plt.subplots(
    nrows=2, sharex=True, figsize=(7, 4.5),
    gridspec_kw={"height_ratios": [2, 1]})

ax_preco.plot(petr.index, petr, color="#7B2FBE")
ax_preco.set_title("PETR4 em 2024 (R$)", loc="left")
ax_preco.grid(axis="y")

ax_ret.bar(retorno.index, retorno * 100, color="#6E6E73")
ax_ret.axhline(0, color="black", linewidth=0.6)
ax_ret.set_title("Retorno diário (%)", loc="left")
pior = retorno.idxmin()                   # anota o pior dia
ax_ret.annotate(f"{retorno[pior]:.1%}",
                xy=(pior, retorno[pior] * 100),
                xytext=(8, 0), textcoords="offset points")

for ax in fig.axes:                       # menos tinta sem dado
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig("petr4_2024.pdf")             # formato vetorial
print(type(fig).__name__, "com", len(fig.axes), "Axes;",
      f"pior dia: {pior:%d/%m/%Y}, {retorno[pior]:.2%}")
