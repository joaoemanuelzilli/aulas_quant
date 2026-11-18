"""US$ 1 no à vista e no futuro rolado: WTI e gás, 2008 a 2024."""
import sys; sys.path.insert(0, "..")
from estilo import *
from futuros import fim_de_mes

fig, axs = figura(altura=60, ncols=2)
for ax, (vista, fut, tit) in zip(axs, [("wti", "cl", "Petróleo WTI"),
                                       ("hh", "ng", "Gás natural (Henry Hub)")]):
    m = fim_de_mes([vista, f"{fut}1", f"{fut}2"], fut)
    r_fut = np.log(m[f"{fut}1"] / m[f"{fut}2"].shift()).fillna(0)
    a = m[vista] / m[vista].iloc[0]
    b = np.exp(r_fut.cumsum())
    ax.plot(a.index, a, color=ROXO)
    ax.plot(b.index, b, color=OCRE)
    ax.set_yscale("log")
    ax.set_title(tit)
    rotulo_final(ax, a.index[-1], a.iloc[-1], f"à vista\n{a.iloc[-1]:.2f}".replace(".", ","), ROXO)
    rotulo_final(ax, b.index[-1], b.iloc[-1], f"rolado\n{b.iloc[-1]:.3f}".replace(".", ","), OCRE)
    ax.set_xlim(a.index[0], a.index[-1] + pd.Timedelta(days=1700))
    ax.set_xticks(pd.to_datetime(["2008", "2012", "2016", "2020", "2024"]))
    ax.set_xticklabels(["2008", "2012", "2016", "2020", "2024"])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v:g}".replace(".", ",")))
axs[0].set_ylim(0.04, 3); axs[1].set_ylim(0.002, 3)
axs[0].set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2]); axs[1].set_yticks([0.003, 0.01, 0.03, 0.1, 0.3, 1, 3])
salvar(fig, "cap09/rolagem")
