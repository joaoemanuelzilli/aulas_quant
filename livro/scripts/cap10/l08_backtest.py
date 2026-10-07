import numpy as np
import pandas as pd
from painel import ACOES, FEATS, dev_teste, painel
from modelos import lgbm_ajustado

dev, teste = dev_teste(painel())
m = lgbm_ajustado().fit(dev[FEATS], dev["y"])   # treina uma vez
teste = teste.assign(prob=m.predict_proba(teste[FEATS])[:, 1])

# num dia sem cotação de um papel a posição é mantida (vale o
# último sinal) e o retorno já foi lançado no pregão anterior,
# cujo ret_seg cobre os dois dias
prob = teste.pivot(columns="acao", values="prob").ffill()
ret = teste.pivot(columns="acao", values="ret_seg").fillna(0)
# CDI acumulado entre o pregão t e o pregão seguinte
cdi = pd.read_csv("../../dados/c10_cdi.csv", index_col=0,
                  parse_dates=True)["cdi"]
pregoes = pd.read_csv("../../dados/precos_ajustados.csv",
                      index_col=0, parse_dates=True)[ACOES]
pregoes = pregoes.dropna(how="all").index
fator = (1 + cdi).cumprod().shift(1, fill_value=1.0)
fator = fator.reindex(pregoes)
cdi = (fator.shift(-1) / fator - 1).reindex(ret.index)
n = prob.shape[1]
CUSTO = 0.001          # custo por ponta: 0,10% do valor negociado


def resultado(pesos):
    caixa = 1 - pesos.sum(axis=1)           # rende o CDI
    giro = pesos.diff().abs().sum(axis=1)
    giro.iloc[0] = pesos.iloc[0].abs().sum()
    bruto = (pesos * ret).sum(axis=1) + caixa * cdi
    liq = bruto - CUSTO * giro
    curva = (1 + liq).cumprod()
    exc = liq - cdi                         # excesso sobre o CDI
    anos = len(liq) / 252
    return {"retorno a.a.": curva.iloc[-1] ** (1 / anos) - 1,
            "sem custos": (1 + bruto).prod() ** (1 / anos) - 1,
            "vol a.a.": liq.std() * np.sqrt(252),
            "Sharpe": exc.mean() / exc.std() * np.sqrt(252),
            "drawdown máx.": (curva / curva.cummax() - 1).min(),
            "exposição": pesos.sum(axis=1).mean(),
            "giro a.a.": giro.mean() * 252}


pesos = (prob > 0.5) / n
tab = pd.DataFrame({
    "modelo": resultado(pesos),
    "com atraso": resultado(pesos.shift(1).fillna(0)),
    "comprar tudo": resultado(prob.notna() / n),
})
print(f"{ret.index[0]:%d/%m/%Y} a {ret.index[-1]:%d/%m/%Y},"
      f" {len(ret)} pregões | CDI: "
      f"{(1 + cdi).prod() ** (252 / len(cdi)) - 1:.1%} a.a.")
print(tab.round(3).to_string())
