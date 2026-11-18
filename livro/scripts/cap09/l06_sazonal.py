"""Sazonalidade do gás natural e volatilidade por vencimento."""
import numpy as np
import pandas as pd
from futuros import FIM_FUTUROS, fim_de_mes, ler

MESES = "jan fev mar abr mai jun jul ago set out nov dez".split()


def spread_por_mes():
    """Média de ln(F2/F1) do gás por mês do pregão, sem os três
    últimos pregões de cada mês (quando o contrato 1 muda)."""
    f = ler(["ng1", "ng2"], "2008", FIM_FUTUROS).dropna()
    meses = f.index.to_period("M")
    resto = f.groupby(meses).cumcount(ascending=False)
    f = f[resto >= 3]
    s = np.log(f["ng2"] / f["ng1"])
    return s.groupby(s.index.month).mean()


def vol_por_mes(coluna="hh"):
    """Volatilidade anualizada do preço à vista em cada mês do ano:
    mediana, entre os anos, do desvio-padrão diário do mês."""
    p = ler([coluna], "2008").dropna()[coluna]
    r = np.log(p[p > 0]).diff().dropna()
    dp = r.groupby([r.index.year, r.index.month]).std()
    return dp.groupby(level=1).median() * np.sqrt(252)


def vol_por_prazo(fut):
    """Volatilidade anualizada do log-retorno mensal do contrato que
    passa de k+1 a k, para k = 1, 2, 3 (efeito Samuelson)."""
    m = fim_de_mes([f"{fut}{k}" for k in range(1, 5)], fut)
    return pd.Series({k: np.log(m[f"{fut}{k}"]
                                / m[f"{fut}{k + 1}"].shift()).std()
                      * np.sqrt(12) for k in (1, 2, 3)})


if __name__ == "__main__":
    sp, vol = spread_por_mes(), vol_por_mes()
    tab = pd.DataFrame({"ln(F2/F1)": sp, "vol HH": vol})
    tab.index = MESES
    print(tab.round(3).T.to_string())
    for fut in ("cl", "ng"):
        v = vol_por_prazo(fut)
        print(fut, "vol por prazo:", v.round(3).tolist())
