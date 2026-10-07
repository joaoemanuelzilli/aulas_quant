"""Resíduos padronizados do GARCH(1,1): quantis contra a normal e contra a t de Student."""
import sys; sys.path.insert(0, "..")
from estilo import *
from arch import arch_model
from scipy import stats

r = 100 * np.log(precos("^BVSP")["^BVSP"].dropna()).diff().dropna()
n = len(r)
pp = (np.arange(1, n + 1) - 0.5) / n
mn = arch_model(r, vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
mt = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
nu = mt.params["nu"]
zn, zt = np.sort(mn.std_resid), np.sort(mt.std_resid)
qn = stats.norm.ppf(pp)
qt = stats.t.ppf(pp, nu) * np.sqrt((nu - 2) / nu)      # t padronizada para variância 1
print(f"nu = {nu:.2f}; menor z (normal) {zn[0]:.2f} em {mn.std_resid.idxmin():%d/%m/%Y}; quantil normal correspondente {qn[0]:.2f}; quantil t {qt[0]:.2f}")
print(f"quantil empírico de 1%: {np.quantile(zn, 0.01):.2f} (normal: {stats.norm.ppf(0.01):.2f}; t padronizada: {stats.t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu):.2f})")
d = mn.std_resid.idxmin()
print(f"em {d:%d/%m/%Y}: retorno {r[d]:.2f}%, vol. condicional do dia {mn.conditional_volatility[d]:.2f}%")
print(f"assimetria dos resíduos padronizados: {stats.skew(zn):.2f}; maior z: {zn[-1]:.2f}")

fig, (a, b) = figura(altura=57, ncols=2, sharey=True)
for ax, q, z, tit in ((a, qn, zn, "Contra a normal"), (b, qt, zt, f"Contra a $t$ com {nu:.1f} graus".replace(".", ","))):
    ax.plot([-6.5, 6.5], [-6.5, 6.5], color=CINZA, lw=0.7)
    ax.plot(q, z, ".", color=ROXO, ms=2.5)
    ax.set_title(tit); ax.set_xlabel("quantil teórico")
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-8, 6.5); ax.set_xticks([-6, -3, 0, 3, 6])
    ax.grid(False)
a.set_ylabel("quantil dos resíduos padronizados")
salvar(fig, "cap04/qq")
