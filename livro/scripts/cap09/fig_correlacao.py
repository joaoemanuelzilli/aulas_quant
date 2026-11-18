"""WTI, Brent e gás: níveis, correlação móvel dos retornos e diferença."""
import sys; sys.path.insert(0, "..")
from estilo import *
from futuros import ler

p = ler(["wti", "brent", "hh"], "2008").dropna()
p = p[(p > 0).all(axis=1)]
sem = p.resample("W-FRI").last().dropna()
r = np.log(sem).diff().dropna()
fig, (a, b, c) = figura(altura=54, largura="larga", ncols=3)
base = sem / sem.iloc[0]
for col, cor, nome in [("wti", ROXO, "WTI"), ("brent", OCRE, "Brent"), ("hh", VERDE, "gás")]:
    a.plot(base.index, base[col], color=cor, lw=0.9)
a.set_yscale("log"); a.set_title("Preço (1 = início de 2008)")
a.set_yticks([0.2, 0.5, 1, 2]); a.set_ylim(0.12, 3.6); a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
a.text(pd.Timestamp("2009-01-01"), 2.3, "WTI e Brent", fontsize=7)
a.text(pd.Timestamp("2012-01-01"), 0.21, "gás", fontsize=7)
for col, cor, nome in [("brent", OCRE, "WTI x Brent"), ("hh", VERDE, "WTI x gás")]:
    rc = r.wti.rolling(52).corr(r[col])
    b.plot(rc.index, rc, color=cor, lw=0.9)
    rotulo_final(b, rc.index[-1], rc.iloc[-1], nome.split(" x ")[1], cor)
b.axhline(0, color=CINZA, lw=0.5)
b.set_title("Correlação móvel com o WTI")
b.set_ylim(-0.5, 1.05); virgula(b)
d = (p.brent - p.wti)
c.plot(d.index, d, color=ROXO, lw=0.6)
c.axhline(0, color=CINZA, lw=0.5)
c.set_title("Brent − WTI (US$ por barril)")
virgula(c, casas=0)
for ax in (a, b, c):
    ax.set_xticks(pd.to_datetime(["2008", "2014", "2020", "2026"]))
    ax.set_xticklabels(["2008", "2014", "2020", "2026"])
b.set_xlim(right=pd.Timestamp("2029-01-01"))
salvar(fig, "cap09/correlacao")
print("Brent-WTI: máx", d.max().round(2), d.idxmax().date(), " média 2008-2010", d[:"2010"].mean().round(2),
      " média 2011-2013", d["2011":"2013"].mean().round(2), " média 2025", d["2025"].mean().round(2),
      " set/2026", d["2026-09"].mean().round(2))
print("corr móvel WTI x Brent: mín", r.wti.rolling(52).corr(r.brent).min().round(2))
