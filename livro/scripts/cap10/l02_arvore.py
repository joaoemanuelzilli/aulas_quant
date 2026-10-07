import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from painel import FEATS, dev_teste, painel

dev, _ = dev_teste(painel())
X, y = dev[FEATS], dev["y"]

rasa = DecisionTreeClassifier(max_depth=2, random_state=0)
rasa.fit(X, y)
print(export_text(rasa, feature_names=FEATS, decimals=3,
                  show_weights=True))

# a mesma árvore em cinco reamostragens dos dias de treino
rng = np.random.default_rng(0)
dias = dev.index.unique()
for b in range(5):
    sorteio = rng.choice(dias, size=len(dias), replace=True)
    a = dev.loc[sorteio]
    t = DecisionTreeClassifier(max_depth=2, random_state=0)
    t.fit(a[FEATS], a["y"])
    print(f"amostra {b}: raiz divide em "
          f"{FEATS[t.tree_.feature[0]]} <= "
          f"{t.tree_.threshold[0]:.3f}")
