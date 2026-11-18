import numpy as np
import pandas as pd
from features import carregar


def tripla_barreira(p, sigma, h=10, b=1.0):
    """Para cada dia t: rótulo (+1 alvo, -1 stop, 0 tempo) e
    número de dias até a primeira barreira tocada."""
    lp = np.log(p.to_numpy())
    larg = b * np.sqrt(h) * sigma.to_numpy()  # largura em t
    rot = np.full(len(lp), np.nan)
    dias = np.full(len(lp), np.nan)
    for t in range(len(lp) - h):         # exige h dias de futuro
        if np.isnan(larg[t]):
            continue
        caminho = lp[t + 1:t + h + 1] - lp[t]
        toque = np.flatnonzero(np.abs(caminho) >= larg[t])
        if len(toque) > 0:               # barreira horizontal
            rot[t] = np.sign(caminho[toque[0]])
            dias[t] = toque[0] + 1
        else:                            # barreira vertical
            rot[t], dias[t] = 0, h
    return pd.DataFrame({"rotulo": rot, "dias": dias},
                        index=p.index)


if __name__ == "__main__":
    d = carregar()
    r = np.log(d["Close"]).diff()
    sigma = r.rolling(21).std()          # vol. diária até t
    b = tripla_barreira(d["Close"], sigma).dropna()
    g = b.groupby("rotulo")["dias"]
    tab = pd.DataFrame({"fração": g.size() / len(b),
                        "dias até o toque": g.mean()})
    print(tab.round(3).to_string())
    fixo = np.sign(np.log(d["Close"]).diff(10).shift(-10))
    fixo = fixo.reindex(b.index)
    contra = (b["rotulo"] * fixo == -1).sum()
    print(f"barreira tocada e retorno de 10 dias com sinal "
          f"oposto: {contra} de {(b['rotulo'] != 0).sum()}")
    print("sem rótulo no fim da amostra:",
          d.index[-10].date(), "a", d.index[-1].date())
    # referência: passeio aleatório gaussiano, mesmas regras
    rng = np.random.default_rng(0)
    ra = pd.Series(rng.normal(0, 0.02, 100_000))
    pa = np.exp(ra.cumsum())
    ba = tripla_barreira(pa, ra.rolling(21).std()).dropna()
    fr = ba["rotulo"].value_counts(normalize=True).sort_index()
    print("passeio aleatório:", fr.round(3).to_dict())
