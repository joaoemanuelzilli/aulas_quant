"""Razão de hedge de mínima variância, dentro e fora da amostra"""
import numpy as np
from futuros import fim_de_mes

m = fim_de_mes(["wti", "brent", "cl1", "cl2"], "cl")
r_fut = (m["cl1"] / m["cl2"].shift() - 1).dropna()  # rolado
treino, teste = slice("2008", "2015"), slice("2016", "2024")

def razao(r_vista, r_fut):
    """h* = Cov(vista, futuro) / Var(futuro), beta da regressão"""
    c = np.cov(r_vista, r_fut)
    return c[0, 1] / c[1, 1]

def reducao(r_vista, r_fut, h):
    """Fração da variância da posição à vista que o hedge tira."""
    return 1 - np.var(r_vista - h * r_fut) / np.var(r_vista)

if __name__ == "__main__":
    for vista in ["wti", "brent"]:
        r = (m[vista] / m[vista].shift() - 1).dropna()
        h = razao(r[treino], r_fut[treino])
        rho = np.corrcoef(r[treino], r_fut[treino])[0, 1]
        print(f"{vista}: h* = {h:.3f} no treino (correlação "
              f"{rho:.3f}); redução da variância:")
        for nome, per in [("treino", treino), ("teste ", teste)]:
            a, b = r[per], r_fut[per]
            print(f"  {nome} ({len(a)} meses): com h* "
                  f"{reducao(a, b, h):.1%}, com h = 1 "
                  f"{reducao(a, b, 1):.1%}; h* do período "
                  f"{razao(a, b):.3f}")
