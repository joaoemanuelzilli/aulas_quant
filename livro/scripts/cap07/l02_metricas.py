import pandas as pd

import metricas as m
from motor import carregar, cruzamento

preco, ret = carregar("PETR4.SA")
bt = cruzamento(preco, ret["cdi"], rapida=20, lenta=100)
bt, ret = bt.loc["2011":"2025"], ret.loc["2011":"2025"]
rf, mercado = ret["cdi"], ret["mercado"]

linhas = {}
series = {"estratégia": bt["retorno"], "PETR4": ret["ativo"],
          "Ibovespa": mercado}
for nome, r in series.items():
    beta, alfa = m.beta_alfa(r, mercado, rf)
    linhas[nome] = {
        "retorno a.a.": f"{m.retorno_anual(r):.1%}",
        "vol a.a.": f"{m.vol_anual(r):.1%}",
        "excesso a.a.": f"{(r - rf).mean() * 252:.1%}",
        "drawdown máx.": f"{m.drawdown_maximo(r):.1%}",
        "Sharpe": f"{m.sharpe(r, rf):.2f}",
        "Sortino": f"{m.sortino(r, rf):.2f}",
        "Calmar": f"{m.calmar(r):.2f}",
        "beta": f"{beta:.2f}",
        "Treynor": f"{m.treynor(r, mercado, rf):.1%}",
        # + 0.0 troca o zero negativo por zero
        "alfa a.a.": f"{round(alfa, 4) + 0.0:.1%}",
    }
print(pd.DataFrame(linhas).to_string())
anos = (rf.index[-1] - rf.index[0]).days / 365.25
print(f"CDI a.a.: {m.retorno_anual(rf):.1%} (252 dias)   "
      f"{(1 + rf).prod() ** (1 / anos) - 1:.1%} (calendário)")
# média aritmética anualizada: o que entra no Sharpe
arit = bt["retorno"].mean() * 252
print(f"média diária x 252: estratégia {arit:.1%}   "
      f"CDI {rf.mean() * 252:.1%}")
ir = m.information_ratio(bt["retorno"], ret["ativo"])
print(f"information ratio da estratégia contra PETR4: {ir:.2f}")
