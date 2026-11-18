"""Motor vetorizado com caixa a zero (comparar ao backtrader)."""
import metricas as m
from motor import carregar, cruzamento

preco, ret = carregar("PETR4.SA")
for defasagem in [1, 2]:
    bt = cruzamento(preco, 0.0 * ret["cdi"], defasagem=defasagem)
    r = bt["retorno"].loc["2011":"2025"]
    print(f"vetorizado, shift({defasagem}): retorno a.a. "
          f"{m.retorno_anual(r):.1%}   "
          f"drawdown máx. {m.drawdown_maximo(r):.1%}")
