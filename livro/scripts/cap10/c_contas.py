"""Confere as contas feitas à mão no texto."""
import numpy as np
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier
from painel import FEATS, dev_teste, painel

dev, _ = dev_teste(painel())
t = DecisionTreeClassifier(max_depth=2, random_state=0)
t.fit(dev[FEATS], dev["y"])
tr = t.tree_
print("nós:", tr.n_node_samples, "\nimpureza:", tr.impurity.round(4))
print("valores:", (tr.value[:, 0, :] * tr.n_node_samples[:, None]).round(0))
g = lambda a, b: 1 - (a / (a + b)) ** 2 - (b / (a + b)) ** 2
v = (tr.value[:, 0, :] * tr.n_node_samples[:, None]).round(0)
raiz, esq, dir_ = g(*v[0]), g(*v[1]), g(*v[4])
ne, nd = tr.n_node_samples[1], tr.n_node_samples[4]
pond = (ne * esq + nd * dir_) / (ne + nd)
print("cortes:", [(FEATS[f], round(float(x), 3)) for f, x in
                  zip(tr.feature, tr.threshold) if f >= 0])
print("frações de alta por nó:", (v[:, 1] / v.sum(1)).round(4))
print("gini raiz", raiz, "esq", esq, "dir", dir_, "pond", pond,
      "ganho", raiz - pond)
h = lambda q: -q * np.log2(q) - (1 - q) * np.log2(1 - q)
print("entropia raiz", h(v[0, 1] / v[0].sum()))
print("bootstrap: 1-1/e =", 1 - np.exp(-1),
      " (1-1/n)^n n=1000:", (1 - 1 / 1000) ** 1000)
for rho, s2, B in ((0.5, 1, 300), (0.1, 1, 300), (0.5, 1, 1)):
    print("var média", rho, B, rho * s2 + (1 - rho) * s2 / B)
for p in (0.52, 0.55, 0.60, 0.70):
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    print("aposta", p, "z", round(z, 4), "m", round(2 * norm.cdf(z) - 1, 4),
          "linear", round(2 * p - 1, 2))
# simulação da variância da média de B variáveis equicorrelacionadas
rng = np.random.default_rng(0)
B, rho = 50, 0.3
C = rho * np.ones((B, B)) + (1 - rho) * np.eye(B)
x = rng.multivariate_normal(np.zeros(B), C, size=200000).mean(1)
print("simulada", x.var(), "fórmula", rho + (1 - rho) / B)
# números citados na seção de discussão
import pandas as pd
print("dias no desenvolvimento:", dev.index.nunique(),
      "| períodos de 5 dias sem sobreposição:", dev.index.nunique() // 5)
c = dev[FEATS].corr()
print("corr pm_50/ret_21", round(c.loc["pm_50", "ret_21"], 2),
      "pm_50/rsi_14", round(c.loc["pm_50", "rsi_14"], 2),
      "ret_21/rsi_14", round(c.loc["ret_21", "rsi_14"], 2))
fw = dev.pivot(columns="acao", values="fwd").corr().to_numpy()
print("corr média entre ações do retorno de 5 dias:",
      round(fw[np.triu_indices(7, 1)].mean(), 3))
d = painel()
print("linhas:", len(d), d.index[0].date(), d.index[-1].date())
print(d.groupby("acao").apply(
    lambda g: (g.index[0].date(), len(g))).to_string())
print("linhas por dia:", d.groupby(level=0).size()
      .value_counts().to_dict())
cont = dev.groupby(level=0).size()
print("dias do desenvolvimento com menos de 7 ações:",
      int((cont < 7).sum()))
folha = dev[t.apply(dev[FEATS]) == 3]      # a folha de 61,7%
print("folha 3:", len(folha), "altas", int(folha["y"].sum()),
      "| ações:", folha["acao"].value_counts().to_dict())
ano = folha.groupby(folha.index.year)["y"].agg(["size", "mean"])
print(ano.round(2).T.to_string())
sel = ano.loc[[2016, 2017, 2018, 2020], "size"].sum()
print("fração em 2016-2018 e 2020:", round(sel / len(folha), 3),
      "| MGLU3:", round((folha["acao"] == "MGLU3.SA").mean(), 3),
      "| dias distintos:", folha.index.nunique())
