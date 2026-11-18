from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, TimeSeriesSplit,
                                     cross_val_score)
import numpy as np
from sklearn.neighbors import (KNeighborsClassifier,
                               NearestNeighbors)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import base

H = 20                            # rótulo: sinal de 20 dias
X, ret = base(h=H)
y = (ret > 0).astype(int)

modelos = {
    "logística": LogisticRegression(),
    "15 vizinhos": KNeighborsClassifier(n_neighbors=15),
}
esquemas = {
    "k-fold embaralhado": KFold(5, shuffle=True, random_state=0),
    "k-fold em blocos": KFold(5),
    "TimeSeriesSplit": TimeSeriesSplit(5),
    "TimeSeriesSplit + gap": TimeSeriesSplit(5, gap=H - 1),
}
print(f"{len(y)} observações, {y.mean():.1%} de altas")
for nome_m, m in modelos.items():
    modelo = make_pipeline(StandardScaler(), m)
    for nome_e, cv in esquemas.items():
        auc = cross_val_score(modelo, X, y, cv=cv,
                              scoring="roc_auc")
        print(f"{nome_m:12s} {nome_e:22s} AUC {auc.mean():.3f}"
              f"  (dobras: {auc.min():.3f} a {auc.max():.3f})")

# diagnóstico: em cada dobra embaralhada, onde estão no
# calendário os 15 vizinhos de treino de cada dia de teste?
dist = []
for tr, te in esquemas["k-fold embaralhado"].split(X):
    pad = StandardScaler().fit(X.iloc[tr])   # só com o treino
    busca = NearestNeighbors(n_neighbors=15)
    busca.fit(pad.transform(X.iloc[tr]))
    _, viz = busca.kneighbors(pad.transform(X.iloc[te]))
    dist.append(np.abs(tr[viz] - te[:, None]))
dist = np.vstack(dist)
perto = (dist < H).sum(axis=1).mean()
acaso = 15 * 2 * (H - 1) / len(X)
print(f"vizinhos a menos de {H} pregões: {perto:.2f} de 15 "
      f"(ao acaso: {acaso:.2f})")
print(f"vizinhos a 1 pregão: "
      f"{(dist <= 1).sum(axis=1).mean():.2f} de 15")
print(f"distância mediana: {np.median(dist):.0f} pregões")
