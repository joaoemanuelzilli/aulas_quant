"""Retornos do Ibovespa e volatilidade condicional do GARCH(1,1)."""
import sys; sys.path.insert(0, "..")
from estilo import *
from arch import arch_model

r = 100 * np.log(precos("^BVSP")["^BVSP"].dropna()).diff().dropna()
res = arch_model(r, vol="GARCH", p=1, q=1).fit(disp="off")
vol = res.conditional_volatility * np.sqrt(252)
w, al, be = res.params[["omega", "alpha[1]", "beta[1]"]]
lp = np.sqrt(252 * w / (1 - al - be))
print(f"longo prazo {lp:.1f}%  máx {vol.max():.1f}% em {vol.idxmax():%d/%m/%Y}  mín {vol.min():.1f}% em {vol.idxmin():%d/%m/%Y}")
print(f"mediana {vol.median():.1f}%  dias acima do longo prazo: {np.mean(vol > lp):.1%}")
for ano in (2011, 2015, 2017, 2020, 2024):
    print(ano, f"vol. condicional média {vol[str(ano)].mean():.1f}%  dp realizado anualizado {r[str(ano)].std() * np.sqrt(252):.1f}%")
print("pior retorno:", r.min().round(2), r.idxmin().date(), " melhor:", r.max().round(2), r.idxmax().date())

fig, (a, b) = figura(altura=75, nrows=2, sharex=True)
a.plot(r.index, r, color=ROXO, lw=0.45)
a.set_title("Log-retorno diário do Ibovespa (%)")
a.set_ylim(-17, 15)
b.plot(vol.index, vol, color=ROXO, lw=0.8)
b.axhline(lp, color=CINZA, lw=0.8, ls=(0, (4, 3)))
b.set_title("Volatilidade condicional do GARCH(1,1), anualizada (%)")
b.text(pd.Timestamp("2021-06-01"), lp + 30, f"nível de longo prazo: {lp:.1f}%".replace(".", ","), fontsize=7, color=CINZA)
b.plot([pd.Timestamp("2022-06-01")] * 2, [lp + 2, lp + 26], color=CINZA, lw=0.5)
b.annotate("março de 2020", (vol.idxmax(), vol.max()), xytext=(-8, -2), textcoords="offset points", ha="right", va="center", fontsize=7)
b.set_ylim(0, 132); b.set_xlim(r.index[0], r.index[-1])
salvar(fig, "cap04/volcond")
