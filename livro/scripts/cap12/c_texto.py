"""Números citados no texto do Capítulo 12 que não saem das
listagens impressas."""
import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import base, carregar, montar

d = carregar()
X0 = montar(d).dropna()
print("primeira linha com todas as features:",
      X0.index[0].date(), "| posição", d.index.get_loc(X0.index[0]))

# logística sem a penalização padrão (C muito grande)
X, ret = base(h=1)
y = (ret > 0).astype(int)
tr = X.index < "2019"
for C in (1.0, 1e6):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C))
    m.fit(X[tr], y[tr])
    print(f"C={C:g}: coeficiente do RSI "
          f"{m[-1].coef_[0][X.columns.get_loc('rsi_14')]:.3f}")

# backtest: regra contra a mistura fixa, e pontas do placebo
with contextlib.redirect_stdout(io.StringIO()):
    import l07_backtest as b


def aa(r):
    return (1 + r).prod() ** (252 / len(r)) - 1


fixo = pd.Series(b.w.mean(), index=b.w.index)
mist = aa(b.estrategia(fixo, b.CUSTO))
print(f"regra sem custo menos mistura: "
      f"{(aa(b.bruto) - mist) * 100:.2f} pontos ao ano")
print(f"mistura menos regra com custo: "
      f"{(mist - aa(b.liquido)) * 100:.2f} pontos ao ano")
rng = np.random.default_rng(0)       # mesmos sorteios do script
pontas = []
for _ in range(2000):
    salto = rng.integers(50, len(b.w) - 50)
    wp = pd.Series(np.roll(b.w.to_numpy(), salto), index=b.w.index)
    pontas.append(wp.diff().abs().fillna(wp.iloc[0]).sum())
print(f"pontas no placebo: {min(pontas):.0f} a {max(pontas):.0f}")
