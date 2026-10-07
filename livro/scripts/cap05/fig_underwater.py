"""Cota da carteira, high-water mark e série de drawdown."""
import sys; sys.path.insert(0, "..")
from estilo import *
from carteira import cota

hwm = cota.cummax()
dd = cota / hwm - 1
fig, (a1, a2) = figura(altura=70, largura="larga", nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
a1.plot(hwm, color=CINZA, lw=0.9)
a1.plot(cota, color=ROXO, lw=1)
a1.set_yscale("log"); a1.set_yticks([1, 2, 5, 10]); a1.set_yticklabels(["1", "2", "5", "10"]); a1.minorticks_off()
a1.set_title("Valor de R$ 1 aplicado na carteira em jan/2010 (escala log) e seu máximo anterior")
a1.text(pd.Timestamp("2015-09-01"), 2.25, "high-water mark", fontsize=7, color=CINZA)
a1.text(pd.Timestamp("2016-09-01"), 1.15, "cota", fontsize=7)
a2.fill_between(dd.index, dd, 0, color=ROXO, alpha=0.85, lw=0)
a2.set_title("Drawdown: distância até o máximo anterior")
pct(a2)
def rot(d):
    return f"{d:%d/%m/%Y}"
episodios = []
for ini, fim in (("2010", "2019"), ("2020", "2020"), ("2021", "2025")):
    vale = dd[ini:fim].idxmin(); pico = cota[:vale].idxmax()
    dep = cota[vale:]; volta = dep[dep >= cota[pico]].index
    episodios.append((pico, vale, volta[0] if len(volta) else None, dd[vale]))
    a2.annotate(f"{dd[vale]:.1%}".replace(".", ",") + f"\n{rot(vale)}", (vale, dd[vale]), xytext=(10, -2), textcoords="offset points",
                fontsize=6.8, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=CINZA))
a2.set_ylim(-0.5, 0.01); a2.set_xlim(dd.index[0], dd.index[-1])
salvar(fig, "cap05/underwater")
for p, v, r, d in episodios:
    print(f"pico {rot(p)} vale {rot(v)} volta {rot(r) if r is not None else '-'} dd {d:.1%}",
          "pregões queda", len(cota[p:v]) - 1, "recup", (len(cota[v:r]) - 1) if r is not None else "-")
print("cota final", cota.iloc[-1], "último dd", dd.iloc[-1])
