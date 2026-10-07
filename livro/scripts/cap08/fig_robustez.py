"""Robustez: Sharpe em função de n (pico estreito) e walk-forward em outros ativos."""
import sys; sys.path.insert(0, "..")
from estilo import *

sn = pd.read_csv(DADOS / "c08_robustez_n.csv")
at = pd.read_csv(DADOS / "c08_robustez_ativos.csv").sort_values("wf").reset_index(drop=True)

fig, (a1, a2) = figura(altura=60, ncols=2, gridspec_kw={"width_ratios": [1.15, 1]})
a1.axhline(0, color=TINTA, lw=0.6)
a1.plot(sn["n"], sn["sr_tudo"], "-o", color=ROXO, ms=3.2, lw=1.1)
a1.set_title("PETR4: Sharpe para cada n fixo"); a1.set_xlabel("n (dias do canal de entrada)")
a1.set_xticks(range(10, 101, 10)); a1.set_ylim(-0.12, 0.66)
a1.annotate("pico em n = 15 e 20", (20, sn.set_index("n").loc[20, "sr_tudo"]), xytext=(28, 0), textcoords="offset points",
            fontsize=7, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
virgula(a1, casas=1)
y = np.arange(len(at))
a2.vlines(0, -0.6, len(at) - 0.4, color=TINTA, lw=0.6)   # não atravessa a legenda
a2.hlines(y, at["comprado"], at["wf"], color=CINZA_CLARO, lw=1.6)
a2.plot(at["comprado"], y, "o", color=OCRE, ms=4.2, label="ativo comprado")
a2.plot(at["wf"], y, "o", color=ROXO, ms=4.2, label="walk-forward")
a2.set_yticks(y); a2.set_yticklabels(at["ativo"]); a2.grid(False); a2.set_ylim(-0.6, len(at) + 0.9)
a2.set_title("Sharpe por ativo, 2013–2025")
a2.legend(loc="upper left", ncols=2, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.0, fontsize=6.8)
a2.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
virgula(a2, "x", casas=1)
salvar(fig, "cap08/robustez")
