import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import base, carregar

CUSTO = 0.0010                    # custo por ponta: 0,10%
X, ret = base(h=1)
R = np.exp(ret) - 1               # retorno simples de t a t+1
y = (ret > 0).astype(int)
cdi = pd.read_csv("../../dados/c09_cdi.csv", index_col=0,
                  parse_dates=True)["cdi"]
# CDI acumulado entre o pregão t e o pregão seguinte
fator = (1 + cdi).cumprod().shift(1, fill_value=1.0)
fator = fator.reindex(carregar().index)
rf = (fator.shift(-1) / fator - 1).reindex(X.index)

prob = pd.Series(np.nan, index=X.index)
for ano in range(2019, 2026):     # reajusta no início de cada ano
    treino, teste = X.index.year < ano, X.index.year == ano
    m = make_pipeline(StandardScaler(), LogisticRegression())
    m.fit(X[treino], y[treino])
    prob[teste] = m.predict_proba(X[teste])[:, 1]

prob = prob.dropna()
w = (prob > 0.5).astype(float)    # 1 = comprado, 0 = no CDI
R, rf = R[prob.index], rf[prob.index]


def estrategia(w, custo):
    giro = w.diff().abs().fillna(w.iloc[0])
    return w * R + (1 - w) * rf - custo * giro


bruto = estrategia(w, 0.0)
liquido = estrategia(w, CUSTO)
atraso = estrategia(w.shift(1).fillna(0), CUSTO)


def resumo(nome, r):
    anos = len(r) / 252
    total = (1 + r).prod()
    exc = r - rf
    sharpe = exc.mean() / exc.std() * np.sqrt(252)
    curva = (1 + r).cumprod()
    dd = (curva / curva.cummax() - 1).min()
    print(f"{nome:20s} {total ** (1 / anos) - 1:6.1%} a.a."
          f"  Sharpe {sharpe:5.2f}  DD máx. {dd:6.1%}")


resumo("modelo, sem custo", bruto)
resumo("modelo, com custo", liquido)
resumo("com custo e atraso", atraso)
resumo("comprar e manter", R)
rf_aa = (1 + rf).prod() ** (252 / len(rf)) - 1
print(f"CDI no período: {rf_aa:.1%} a.a.")
giro = w.diff().abs().fillna(w.iloc[0])
print(f"dias comprado: {w.mean():.1%} | pontas: "
      f"{int(giro.sum())} | acerto do sinal: "
      f"{((prob > 0.5) == (y[prob.index] == 1)).mean():.1%}")
