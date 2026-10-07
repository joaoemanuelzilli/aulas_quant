"""Volatilidade das carteiras quando as correlações sobem."""
import sys; sys.path.insert(0, "..")
from estilo import *
from carteira import ACOES, retornos, vol

hoje = pd.read_csv(DADOS / "c06_carteiras_hoje.csv", index_col=0)
cov = retornos("2023", "2025").cov().values * 252
dp = np.sqrt(np.diag(cov))
corr = cov / np.outer(dp, dp)
fora = ~np.eye(len(dp), dtype=bool)
lams = np.linspace(0, 1, 51)
media = [((1 - l) * corr + l)[fora].mean() for l in lams]
fig, ax = figura(altura=57)
for nome, cor in zip(hoje.columns, CATEGORICAS):
    v = [vol(hoje[nome].values,
             ((1 - l) * corr + l * np.ones_like(corr)) * np.outer(dp, dp))
         for l in lams]
    ax.plot(media, v, color=cor)
    desl = {"mín. var.": 4, "máx. Sharpe": -4}.get(nome, 0)
    ax.annotate(nome, (media[-1], v[-1]), xytext=(4, desl),
                textcoords="offset points", va="center", fontsize=7.2)
    print(nome, round(v[0], 4), round(v[-1], 4), round(hoje[nome].values @ dp, 4))
covid = retornos("2020-01-24", "2020-03-23").corr().values[fora].mean()
ax.axvline(covid, color=CINZA, lw=0.7, ls=(0, (4, 3)))
ax.text(covid - 0.012, 0.118, "correlação média\nobservada na Covid-19",
        ha="right", fontsize=6.8, color=CINZA)
ax.annotate("correlações de 2023–2025\n(média " + f"{media[0]:.2f})".replace(".", ","),
            (media[0], 0.1788), xytext=(14, 30), textcoords="offset points",
            fontsize=6.8, color=CINZA,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
ax.set_xlim(media[0], 1.17); ax.set_ylim(0.11, 0.32)
ax.spines["bottom"].set_bounds(media[0], 1)
ax.set_xticks([0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.12, 0.16, 0.20, 0.24, 0.28, 0.32])
ax.set_xlabel("correlação média entre as ações")
ax.set_ylabel("volatilidade anual da carteira")
virgula(ax, "x"); pct(ax)
salvar(fig, "cap06/estresse_correlacao")
