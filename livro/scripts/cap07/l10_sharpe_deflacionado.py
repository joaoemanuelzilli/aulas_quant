"""Sharpe deflacionado da campeã da grade (2011 a 2018)."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from motor import carregar, cruzamento
from otimizacao import LENTAS, PERIODOS, RAPIDAS
from sobreajuste import maximo_esperado

preco, ret = carregar("PETR4.SA")
dentro = PERIODOS["dentro"]
rf = ret["cdi"].loc[dentro]
excessos = {}
for rapida in RAPIDAS:
    for lenta in LENTAS:
        if rapida < lenta:
            bt = cruzamento(preco, ret["cdi"], rapida, lenta)
            r = bt["retorno"].loc[dentro]
            excessos[rapida, lenta] = r - rf
excessos = pd.DataFrame(excessos)
sharpes = excessos.mean() / excessos.std()     # por dia
campea = sharpes.idxmax()
d = excessos[campea]
T, sr = len(d), sharpes[campea]
g3, g4 = d.skew(), d.kurt() + 3      # assimetria e curtose


def confianca(referencia):
    """Confiança com que se rejeita Sharpe <= referência."""
    ep = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1))
    return norm.cdf((sr - referencia) / ep)


ano = np.sqrt(252)
print(f"campeã {campea[0]}/{campea[1]}: "
      f"Sharpe {sr * ano:.2f} em {T} pregões; "
      f"assimetria {g3:.2f}, curtose {g4:.1f}")
print(f"desvio-padrão dos {len(sharpes)} Sharpes: "
      f"{sharpes.std() * ano:.2f}")
print(f"referência zero: confiança {confianca(0.0):.1%}")
for n in [10, 90]:
    ref = maximo_esperado(n, sharpes.std())
    print(f"N = {n:2d}: referência {ref * ano:.2f}   "
          f"confiança {confianca(ref):.1%}")
corr = excessos.corr().to_numpy()
media = corr[np.triu_indices_from(corr, k=1)].mean()
print(f"correlação média entre as variantes: {media:.2f}")
