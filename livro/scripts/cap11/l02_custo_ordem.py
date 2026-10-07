import numpy as np
import pandas as pd

D = "../../dados/"
ohlc = pd.read_csv(D + "petr4_ohlcv.csv", index_col=0,
                   parse_dates=True).loc["2025"]
tela = pd.read_csv(D + "precos_sem_ajuste.csv", index_col=0,
                   parse_dates=True)["PETR4.SA"].loc["2025"]

preco = tela.dropna().iloc[-1]             # último fechamento
V = (ohlc.Volume * tela).median()          # volume diário, R$
sigma = ohlc.Close.pct_change().std()      # vol. diária

TAXAS = 3.0                                # pb (hipótese)
MEIO_SPREAD = 0.5 * 0.01 / preco * 1e4     # spread de 1 centavo
Y = 1.0                                    # coef. (hipótese)


def custo_pb(Q):
    """Custo de uma ordem de Q reais, em pontos-base por ponta."""
    pico = Y * sigma * np.sqrt(Q / V) * 1e4    # impacto ao final
    return TAXAS + MEIO_SPREAD, 2 / 3 * pico   # custo médio


def ordem_maxima(equilibrio_pb):
    """Maior ordem que se paga, dado o custo de equilíbrio."""
    fixo, _ = custo_pb(1)
    if equilibrio_pb <= fixo:              # nem a menor ordem
        return 0.0
    pico = (equilibrio_pb - fixo) * 1.5        # inverte custo_pb
    return V * (pico / (Y * sigma * 1e4)) ** 2


if __name__ == "__main__":
    from l03_custos import ATIVOS, resultado
    print(f"preço {preco:.2f}  volume mediano R$ {V/1e6:.0f} mi"
          f"  vol. diária {sigma:.2%}")
    print("   ordem (R$)  % do volume  fixo  impacto  total"
          "  custo (R$)")
    for Q in (1e4, 1e5, 1e6, 1e7, 1e8):
        fixo, imp = custo_pb(Q)
        tot = fixo + imp
        print(f"{Q:13,.0f} {Q / V:11.3%} {fixo:5.1f} {imp:8.1f}"
              f" {tot:6.1f} {Q * tot / 1e4:11,.0f}")
    print("versão  equilíbrio  ordem máxima   % do volume"
          "  capital")
    for nome, (bruto, giro) in resultado.items():
        eq = bruto.sum() / giro.sum() * 1e4
        Qmax = ordem_maxima(eq)
        # fração do capital negociada por papel em cada ajuste
        fracao = giro[giro > 0].mean() / len(ATIVOS)
        capital = Qmax / fracao
        print(f"{nome}  {eq:7.1f} pb  R$ {Qmax / 1e6:6.1f} mi"
              f"  {Qmax / V:11.2%}  R$ {capital / 1e6:,.0f} mi")
