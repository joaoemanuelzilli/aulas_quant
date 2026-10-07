import numpy as np
from painel import FEATS, dev_teste, painel
from modelos import lgbm_ajustado

dev, teste = dev_teste(painel())
m = lgbm_ajustado().fit(dev[FEATS], dev["y"])

x = dev[FEATS].iloc[[-1]]      # última linha do desenvolvimento
phi = m.booster_.predict(x, pred_contrib=True)[0]
escore = m.booster_.predict(x, raw_score=True)[0]
print("valor base (log-chance média):", phi[-1].round(4))
maiores = sorted(zip(FEATS, phi[:-1]), key=lambda t: -abs(t[1]))
for nome, v in maiores[:3]:
    print(f"  {nome:10s} {v:+.4f}")
print("base + contribuições:", phi.sum().round(4),
      "| escore do modelo:", escore.round(4))
print("probabilidade:", (1 / (1 + np.exp(-escore))).round(4))
