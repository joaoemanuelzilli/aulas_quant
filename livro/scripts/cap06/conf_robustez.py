"""Robustez do experimento fora da amostra (números citados na
Seção 6.7 e gabarito do exercício da janela). Leva alguns
minutos. Não grava nada em dados/."""
import numpy as np
import pandas as pd

import experimento
from carteira import (ACOES, NOMES, carregar_cdi, insumos,
                      maximo_sharpe, minima_variancia, retornos,
                      vol)
from experimento import resumo, simular

r = retornos("2012", "2025")
cdi = carregar_cdi(r.index)
REGRAS = ["1/N", "mín. variância", "máx. Sharpe"]


def sharpes(colunas, janela=756):
    experimento.JANELA = janela
    d, g = simular(r[colunas], cdi)
    experimento.JANELA = 756
    t = resumo(d, g, cdi)
    return t.loc[REGRAS, ["Sharpe", "giro/mês"]].round(2), d


# 1. retirar uma ação por vez
print("Sharpe fora da amostra retirando uma ação:")
for fora in ACOES:
    t, _ = sharpes([a for a in ACOES if a != fora])
    print(f"  sem {fora[:5]}:", t["Sharpe"].to_dict())

# 2. tamanho da janela de estimação (exercício)
seis = [a for a in ACOES if a != "MGLU3.SA"]
for janela in [252, 504, 756, 1260]:
    t7, d = sharpes(ACOES, janela)
    t6, _ = sharpes(seis, janela)
    print(f"janela {janela}: início {d.index[0]:%d/%m/%Y}")
    print("  sete ações:", t7["Sharpe"].to_dict(),
          "giro", t7["giro/mês"].to_dict())
    print("  sem MGLU3: ", t6["Sharpe"].to_dict())

# 3. peso de MGLU3 na regra de máximo Sharpe, média por ano,
#    e volatilidade prevista x realizada da mínima variância
meses = r.index.to_period("M")
linhas, prev = {}, {}
for mes in meses.unique():
    hoje = np.argmax(meses == mes)
    if hoje < 756:
        continue
    passado = r.iloc[hoje - 756:hoje]
    mu, cov = insumos(passado)
    rf = cdi.iloc[hoje - 756:hoje].mean() * 252
    linhas[mes] = maximo_sharpe(mu, cov, rf)
    prev[mes] = vol(minima_variancia(cov), cov)
w = pd.DataFrame(linhas, index=NOMES).T
print("peso médio de MGLU3 no máx. Sharpe, por ano (%):")
print((w["MGLU3"].groupby(w.index.year).mean() * 100)
      .round(0).to_dict())
prev = pd.Series(prev)
d = pd.read_csv("../../dados/c06_fora_da_amostra.csv",
                index_col=0, parse_dates=True)
real = d["mín. variância"].std() * np.sqrt(252)
print(f"mín. variância: vol prevista média {prev.mean():.1%} "
      f"(de {prev.min():.1%} a {prev.max():.1%}); "
      f"realizada {real:.1%}")
