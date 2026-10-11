import numpy as np
import pandas as pd

from carteira import (ACOES, NOMES, cdi_anual, insumos,
                      maximo_sharpe, minima_variancia, retornos)

# carteiras "de hoje": estimadas com os três últimos anos
r = retornos("2023", "2025")
mu, cov = insumos(r)
n = len(mu)
hoje = pd.DataFrame(
    {"1/N": np.full(n, 1 / n),
     "mín. var.": minima_variancia(cov),
     "máx. Sharpe": maximo_sharpe(mu, cov,
                                  cdi_anual("2023", "2025"))},
    index=ACOES)

EPISODIOS = {"Covid-19": ("2020-01-23", "2020-03-23"),
             "Joesley Day": ("2017-05-17", "2017-05-18"),
             "fev-jun/2018": ("2018-02-26", "2018-06-18"),
             "2021-2022": ("2021-06-07", "2022-07-14")}
p = pd.read_csv("../../dados/precos_ajustados.csv",
                index_col=0, parse_dates=True)
linhas = {}
for nome, (ini, fim) in EPISODIOS.items():
    choque = p.loc[fim] / p.loc[ini] - 1   # retorno no episódio
    linhas[nome] = choque[ACOES] @ hoje
    linhas[nome]["Ibovespa"] = choque["^BVSP"]
    linhas[nome]["pregões"] = len(p.loc[ini:fim, "PETR4.SA"]
                                  .dropna()) - 1
cenarios = pd.DataFrame(linhas).T
cenarios.to_csv("../../dados/c06_estresse_historico.csv")
hoje.to_csv("../../dados/c06_carteiras_hoje.csv")

print((hoje * 100).round(1).T.set_axis(NOMES, axis=1)
      .to_string())
print()
print((cenarios.drop(columns="pregões") * 100).round(1)
      .assign(pregões=cenarios["pregões"].astype(int))
      .to_string())

# o que o VaR de 1 dia dizia para as mesmas carteiras
diario = r @ hoje
var99 = -diario.quantile(0.01)
print("\nVaR histórico 99%, 1 dia (2023-2025), em %:")
print((var99 * 100).round(2).to_string())
print("escalado pela raiz de 40 pregões, em %:")
print((var99 * np.sqrt(40) * 100).round(1).to_string())
