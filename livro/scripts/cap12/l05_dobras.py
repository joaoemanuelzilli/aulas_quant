import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

X = np.arange(12).reshape(-1, 1)      # 12 observações em ordem
esquemas = {
    "k-fold embaralhado": KFold(n_splits=3, shuffle=True,
                                random_state=0),
    "janela expansível": TimeSeriesSplit(n_splits=3),
    "max_train_size=4": TimeSeriesSplit(n_splits=3,
                                        max_train_size=4),
    "gap=2": TimeSeriesSplit(n_splits=3, gap=2),
}
for nome, cv in esquemas.items():
    print(nome)
    for treino, teste in cv.split(X):
        print("  treino", treino, "| teste", teste)
