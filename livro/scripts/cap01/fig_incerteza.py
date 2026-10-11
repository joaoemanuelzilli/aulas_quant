"""Retorno anualizado de dez anos e a faixa de dois erros-padrão em torno dele."""
import sys; sys.path.insert(0, "..")
from estilo import *

nomes = {"PETR4.SA": "PETR4", "WEGE3.SA": "WEGE3", "VALE3.SA": "VALE3",
         "BOVA11.SA": "BOVA11", "ABEV3.SA": "ABEV3"}
# mesma amostra do Código l04_anualizar.py: base no último fechamento de 2014
p = precos(*nomes, inicio="2014", fim="2024").dropna()
p = p.loc[p.loc["2014"].index[-1]:]
log = np.log(p / p.shift(1)).dropna()
anos = (p.index[-1] - p.index[0]).days / 365.25
r_log = np.log(p.iloc[-1] / p.iloc[0]) / anos          # log-retorno médio anual
ep = log.std() * np.sqrt(252) / np.sqrt(anos)           # erro-padrão dessa média
centro, inf, sup = np.exp(r_log) - 1, np.exp(r_log - 2 * ep) - 1, np.exp(r_log + 2 * ep) - 1

# CDI entre as mesmas datas (a taxa datada de d rende de d a d+1)
cdi = pd.read_csv(DADOS / "c02_bcb.csv", index_col=0, parse_dates=True)["cdi"].dropna() / 100
cdi_aa = (1 + cdi.loc[p.index[0]:p.index[-1]].iloc[:-1]).prod() ** (1 / anos) - 1

fig, ax = figura(altura=52)
for i, t in enumerate(nomes):
    ax.plot([inf[t], sup[t]], [i, i], color=ROXO, lw=1.6, solid_capstyle="butt")
    ax.plot([centro[t]], [i], "o", color=ROXO, ms=4.5)
    ax.annotate(f"{centro[t] * 100:.1f}%".replace(".", ","), (centro[t], i), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=6.8,
                zorder=6, bbox=dict(fc="white", ec="none", pad=1.2))
ax.axvline(0, color=TINTA, lw=0.6, zorder=1)
ax.axvline(cdi_aa, color=CINZA, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.annotate("CDI: " + f"{cdi_aa * 100:.1f}%".replace(".", ",") + " ao ano", (cdi_aa, len(nomes) - 0.45),
            xytext=(3, 0), textcoords="offset points", fontsize=6.8, color=CINZA, va="center")
ax.set_yticks(range(len(nomes))); ax.set_yticklabels(nomes.values(), color=TINTA)
ax.set_ylim(len(nomes) - 0.2, -0.7)
ax.set_xlim(-0.25, 0.85)
pct(ax, "x")
ax.grid(False); ax.grid(True, axis="x")
ax.set_title("Retorno anualizado de 2015 a 2024 e faixa de dois erros-padrão")
salvar(fig, "cap01/incerteza")
print(f"anos {anos:.2f}; CDI {cdi_aa:.2%}")
print(pd.DataFrame({"retorno a.a.": centro, "erro": ep, "de": inf, "a": sup}).map(lambda v: f"{v:.1%}"))
