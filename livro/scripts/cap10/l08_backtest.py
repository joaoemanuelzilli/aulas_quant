import numpy as np
import pandas as pd
from painel import ACOES, FEATS, dev_teste, painel
from modelos import lgbm_ajustado

dev, teste = dev_teste(painel())
m = lgbm_ajustado().fit(dev[FEATS], dev["y"])   # treina uma vez
teste = teste.assign(prob=m.predict_proba(teste[FEATS])[:, 1])

prob = teste.pivot(columns="acao", values="prob")
ret = teste.pivot(columns="acao", values="ret_seg")  # de t a t+1
assert not prob.isna().any().any()   # sete ações em todo pregão
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


def carteira(pesos):
    """Retorno líquido, retorno bruto e giro diários."""
    caixa = 1 - pesos.sum(axis=1)           # rende o CDI
    bruto = (pesos * ret).sum(axis=1) + caixa * cdi
    # peso a que a posição de ontem chegou com os preços de hoje
    antes = (pesos * (1 + ret)).div(1 + bruto, axis=0).shift(1)
    giro = (pesos - antes.fillna(0.0)).abs().sum(axis=1)
    return bruto - CUSTO * giro, bruto, giro


def resultado(pesos):
    liq, bruto, giro = carteira(pesos)
    curva = (1 + liq).cumprod()
    pico = curva.cummax().clip(lower=1.0)   # capital inicial 1
    exc = liq - cdi                         # excesso sobre o CDI
    anos = len(liq) / 252
    return {"retorno a.a.": curva.iloc[-1] ** (1 / anos) - 1,
            "sem custos": (1 + bruto).prod() ** (1 / anos) - 1,
            "vol a.a.": liq.std() * np.sqrt(252),
            "Sharpe": exc.mean() / exc.std() * np.sqrt(252),
            "drawdown máx.": (curva / pico - 1).min(),
            "exposição": pesos.sum(axis=1).mean(),
            "giro a.a.": giro.mean() * 252}


pesos = ((prob > 0.5) / n).astype(float)
todas = (prob.notna() / n).astype(float)
if __name__ == "__main__":
    tab = pd.DataFrame({
        "modelo": resultado(pesos),
        "com atraso": resultado(pesos.shift(1).fillna(0.0)),
        "comprar tudo": resultado(todas),
    })
    print(f"{ret.index[0]:%d/%m/%Y} a {ret.index[-1]:%d/%m/%Y},"
          f" {len(ret)} pregões | CDI: "
          f"{(1 + cdi).prod() ** (252 / len(cdi)) - 1:.1%} a.a.")
    print(tab.round(3).to_string())
