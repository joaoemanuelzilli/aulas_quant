"""Prever o WTI à vista: passeio aleatório, futuro e um modelo."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from futuros import fim_de_mes, ler

fut = np.log(fim_de_mes(["cl1", "cl2", "cl3"], "cl"))
fut.index = fut.index.to_period("M")
vista = ler(["wti"], "2007-12")["wti"].dropna()
s = np.log(vista[vista > 0].resample("ME").last())
s.index = s.index.to_period("M")          # último à vista do mês


def previsoes(h, inicio_teste="2015-01"):
    """Previsões de ln S(m+h) feitas no fim de cada mês m."""
    alvo = s.shift(-h).reindex(fut.index)
    linhas = []
    for m in fut.loc[inicio_teste:].index:
        # só usa pares (origem, alvo) com o alvo já conhecido em m
        passado = fut.loc[:m - h].index
        y = (alvo - s).loc[passado]
        x = (fut[f"cl{h}"] - s).loc[passado]
        b = sm.OLS(y, sm.add_constant(x)).fit().params
        deriva = s.diff().loc[:m].mean() * h
        f = fut.at[m, f"cl{h}"]
        linhas.append({"mes": m, "real": alvo[m], "passeio": s[m],
                       "futuro": f, "deriva": s[m] + deriva,
                       "modelo": s[m] + b.iloc[0]
                       + b.iloc[1] * (f - s[m])})
    return pd.DataFrame(linhas).set_index("mes").dropna()


if __name__ == "__main__":
    print("EQM / EQM do passeio aleatório, teste de 2015 a 2024")
    print("h  previsões  futuro  modelo  deriva   t  sem 2020")
    for h in (1, 2, 3):
        p = previsoes(h)
        e = p.drop(columns="real").sub(p["real"], axis=0)
        eqm = (e ** 2).mean() / (e["passeio"] ** 2).mean()
        d = e["futuro"] ** 2 - e["passeio"] ** 2   # perda
        hac = {"maxlags": h}          # previsões se sobrepõem
        t = sm.OLS(d, np.ones(len(d))).fit(
            cov_type="HAC", cov_kwds=hac).tvalues.iloc[0]
        q = e.drop(p.loc["2020"].index) ** 2
        sem = q["futuro"].mean() / q["passeio"].mean()
        print(f"{h}  {len(p):9d}  {eqm['futuro']:.3f}   "
              f"{eqm['modelo']:.3f}   {eqm['deriva']:.3f}  "
              f"{t:5.2f}  {sem:.3f}")
