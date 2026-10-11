import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ativos = ["PETR4.SA", "VALE3.SA", "ABEV3.SA", "WEGE3.SA",
          "BOVA11.SA"]
p = precos.loc["2014":"2024", ativos].dropna()
p = p.loc[p.loc["2014"].index[-1]:]   # base: fim de 2014
log = np.log(p / p.shift(1)).dropna()

anos = (p.index[-1] - p.index[0]).days / 365.25    # calendário
crescimento = p.iloc[-1] / p.iloc[0]
vol = log.std() * np.sqrt(252)
resumo = pd.DataFrame({
    "retorno total": crescimento - 1,
    "retorno a.a.": crescimento ** (1 / anos) - 1,   # CAGR
    "vol a.a.": vol,
    "erro a.a.": vol / np.sqrt(anos),  # erro-padrão da média
})

# conferência: a soma dos logs refaz o crescimento de cada ativo
assert np.allclose(np.exp(log.sum()), crescimento)

print(f"{p.index[0]:%d/%m/%Y} a {p.index[-1]:%d/%m/%Y}: "
      f"{len(log)} retornos em {anos:.2f} anos")
print(resumo.map(lambda v: f"{v:.1%}"))
