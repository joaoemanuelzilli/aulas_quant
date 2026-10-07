"""Previsão de um AR(1) para o retorno e o leque correspondente para o nível do índice."""
import sys; sys.path.insert(0, "..")
from estilo import *
from statsmodels.tsa.arima.model import ARIMA

ibov = precos("^BVSP")["^BVSP"].dropna()
logp = np.log(ibov)
treino = logp[:"2021"]
H, PASS = 60, 60
real = logp["2022":].iloc[:H]
r_tr = 100 * treino.diff().dropna().to_numpy()

m_r = ARIMA(r_tr, order=(1, 0, 0)).fit()                    # AR(1) nos retornos
f_r = m_r.get_forecast(H); ic_r = f_r.conf_int()
m_p = ARIMA(treino.to_numpy(), order=(1, 1, 0), trend="t").fit()   # o mesmo modelo, escrito no log do preço
f_p = m_p.get_forecast(H); ic_p = f_p.conf_int()
print("phi nos dois ajustes:", m_r.params[1].round(4), m_p.params[1].round(4))
print(f"último retorno do treino: {r_tr[-1]:.2f}%  previsões h=1,2,3,10: {f_r.predicted_mean[[0, 1, 2, 9]].round(4)}  média: {m_r.params[0]:.4f}")
print(f"meia-largura do IC 95% do retorno: h=1 {(ic_r[0, 1] - ic_r[0, 0]) / 2:.3f}  h=60 {(ic_r[-1, 1] - ic_r[-1, 0]) / 2:.3f}  1,96*dp amostral {1.96 * r_tr.std():.3f}")
larg = (ic_p[:, 1] - ic_p[:, 0]) / 2
print(f"meia-largura do IC do log-preço: h=1 {larg[0]:.4f}  h=4 {larg[3]:.4f}  h=16 {larg[15]:.4f}  h=60 {larg[-1]:.4f}  razão h=60/h=1 {larg[-1] / larg[0]:.2f}  raiz(60) {np.sqrt(60):.2f}")
print(f"origem {treino.index[-1]:%d/%m/%Y}: {ibov[treino.index[-1]]:.0f} pontos; IC 95% em h=60: {np.exp(ic_p[-1, 0]):.0f} a {np.exp(ic_p[-1, 1]):.0f}; realizado em {real.index[-1]:%d/%m/%Y}: {np.exp(real.iloc[-1]):.0f}")
dentro = np.mean((real.to_numpy() >= ic_p[:, 0]) & (real.to_numpy() <= ic_p[:, 1]))
print(f"dias do realizado dentro do intervalo: {dentro:.0%}")

fig, (a, b) = figura(altura=58, ncols=2, gridspec_kw={"width_ratios": [1, 1.25]})
h = np.arange(1, H + 1)
r_real = 100 * logp.diff()["2022":].iloc[:H].to_numpy()
a.fill_between(h, ic_r[:, 0], ic_r[:, 1], color=LILAS)
a.plot(h, r_real, ".", color=CINZA, ms=3)
a.plot(h, f_r.predicted_mean, color=ROXO)
a.set_title("Retorno diário (%)"); a.set_xlabel("passos à frente $h$")
a.set_xlim(0, H + 1); a.set_ylim(-4.6, 4.6)
hist = ibov[:"2021"].iloc[-PASS:] / 1000
x0 = np.arange(-PASS + 1, 1)
b.fill_between(h, np.exp(ic_p[:, 0]) / 1000, np.exp(ic_p[:, 1]) / 1000, color=LILAS)
b.plot(x0, hist, color=TINTA, lw=1)
b.plot(h, np.exp(real) / 1000, color=CINZA, lw=1)
b.plot(np.r_[0, h], np.r_[hist.iloc[-1], np.exp(f_p.predicted_mean) / 1000], color=ROXO)
b.set_title("Ibovespa (mil pontos)"); b.set_xlabel("dias úteis desde a origem da previsão")
b.set_xlim(-PASS, H + 1)
b.axvline(0, color=CINZA_CLARO, lw=0.6, zorder=0)
b.text(-57, 123.5, "histórico", fontsize=7)
b.text(33, 121.3, "realizado", fontsize=7, color=CINZA)
b.text(38, 101.2, "previsão", fontsize=7)
salvar(fig, "cap04/previsao")
