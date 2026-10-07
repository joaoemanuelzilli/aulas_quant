"""Volatilidade do mercado e correlação média entre ações, em janelas móveis."""
import sys; sys.path.insert(0, "..")
from estilo import *

acoes = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "ABEV3.SA", "WEGE3.SA", "MGLU3.SA"]
J = 63                                              # um trimestre de pregões
p = precos(*acoes, "^BVSP", inicio="2014-09", fim="2024").dropna()
r = p.pct_change().dropna()
vol = r["^BVSP"].rolling(J).std() * np.sqrt(252)
cm = r[acoes].rolling(J).corr()                     # uma matriz 7 x 7 por dia
n = len(acoes)
media = (cm.groupby(level=0).sum().sum(axis=1) - n) / (n * (n - 1))
media[cm.groupby(level=0).count().min(axis=1) < n] = np.nan
vol, media = vol["2015":], media["2015":]

fig, (a1, a2) = figura(altura=76, nrows=2, sharex=True)
a1.plot(vol.index, vol, color=ROXO)
a1.set_title("Volatilidade anualizada do Ibovespa, janela de 63 pregões")
pct(a1)
a2.plot(media.index, media, color=ROXO)
a2.set_title("Correlação média entre sete ações, mesma janela")
virgula(a2)
a2.set_ylim(0, 0.9)
for ax in (a1, a2):
    ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-30"), color=CINZA_CLARO, alpha=0.6, zorder=0)
a1.text(pd.Timestamp("2020-08-01"), vol.max() * 0.92, "pandemia", fontsize=6.8, color=CINZA)
salvar(fig, "cap03/corr_movel")

print("vol: máx", vol.max().round(3), vol.idxmax().date(), " mediana", vol.median().round(3))
print("corr média: máx", media.max().round(3), media.idxmax().date(), " mín", media.min().round(3),
      media.idxmin().date(), " mediana", media.median().round(3))
print("correlação entre as duas curvas:", vol.corr(media).round(3),
      " Spearman:", vol.corr(media, method="spearman").round(3))
# correlação média por período fixo (sem janela móvel)
def corr_media(x):
    c = x.corr().values
    return c[np.triu_indices(n, 1)].mean()
ra = r[acoes]
print("2015-2024:", corr_media(ra["2015":]).round(3), " fev-abr/2020:", corr_media(ra["2020-02-20":"2020-04-30"]).round(3),
      f"({len(ra['2020-02-20':'2020-04-30'])} dias)", " 2015-2024 sem 2020:", corr_media(ra[ra.index.year != 2020]["2015":]).round(3))
for ano in range(2015, 2025):
    print(ano, corr_media(ra.loc[str(ano)]).round(3), f"vol Ibov {r['^BVSP'][str(ano)].std() * np.sqrt(252):.3f}")
pv = ra["2020-02-20":"2020-04-30"][["PETR4.SA", "VALE3.SA"]].corr().iloc[0, 1]
print("PETR4 x VALE3 fev-abr/2020:", pv.round(3), " 2015-2024:", ra["2015":][["PETR4.SA", "VALE3.SA"]].corr().iloc[0, 1].round(3))
par = r["PETR4.SA"].rolling(J).corr(r["VALE3.SA"])["2015":]
print("PETR4 x VALE3, correlação móvel: máx", par.max().round(3), par.idxmax().date())
