"""Testes de sanidade das funções centrais do capítulo.
Rodar com: python teste_sanidade.py (imprime 'tudo certo')."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from painel import ACOES, FEATS, H, cortes, dev_teste, features, painel

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
merc = precos["BOVA11.SA"].dropna().pct_change(21)

# 1. o futuro não muda o passado: features calculadas com a série
#    truncada coincidem com as da série inteira
p = precos["PETR4.SA"].dropna()
corte = p.index[2500]
m = merc.reindex(p.index)
inteira = features(p, m).loc[:corte]
truncada = features(p.loc[:corte], m.loc[:corte])
pd.testing.assert_frame_equal(inteira, truncada)

# 2. divisões: validação sempre depois do treino, com folga de H
d = painel()
dev, teste = dev_teste(d)
dias = np.sort(dev.index.unique())
for tr, va in cortes(dev.index):
    fim, ini = dev.index[tr].max(), dev.index[va].min()
    assert np.searchsorted(dias, ini) - np.searchsorted(dias, fim) > H
    assert len(np.intersect1d(tr, va)) == 0

# 3. o rótulo do último dia do desenvolvimento não usa preço do teste
for a in ACOES:
    pa = precos[a].dropna()
    ult = dev[dev["acao"] == a].index.max()
    assert pa.index[pa.index.get_loc(ult) + H] < teste.index.min()

# 4. rótulo coerente com o retorno futuro; sem ausentes no painel
assert ((d["fwd"] > 0).astype(int) == d["y"]).all()
assert not d[FEATS + ["fwd", "ret_seg"]].isna().any().any()

# 5. no LightGBM, subsample só age com subsample_freq > 0
X, y = dev[FEATS].to_numpy()[:3000], dev["y"].to_numpy()[:3000]
kw = dict(n_estimators=20, random_state=0, verbose=-1, n_jobs=1)
p0 = LGBMClassifier(**kw).fit(X, y).predict_proba(X)
p1 = LGBMClassifier(subsample=0.5, **kw).fit(X, y).predict_proba(X)
p2 = LGBMClassifier(subsample=0.5, subsample_freq=1, **kw)
p2 = p2.fit(X, y).predict_proba(X)
assert np.allclose(p0, p1) and not np.allclose(p0, p2)

# 6. backtest: carteira sempre em caixa rende exatamente o CDI e
#    comprar tudo sem custo é a média dos retornos das ações
import l08_backtest as bt
liq, bruto, giro = bt.carteira(bt.todas * 0.0)
assert np.allclose(liq, bt.cdi) and giro.sum() == 0
liq, bruto, giro = bt.carteira(bt.todas)
assert np.allclose(bruto, bt.ret.mean(axis=1))
print("tudo certo")
