"""Um pregão de atraso na regra 20/100, em seis ativos."""
import metricas as m
from motor import carregar, cruzamento

for ativo in ["PETR4.SA", "VALE3.SA", "WEGE3.SA", "ABEV3.SA",
              "BBAS3.SA", "BOVA11.SA"]:
    preco, ret = carregar(ativo)
    taxas = []
    for defasagem in [1, 2]:
        bt = cruzamento(preco, ret["cdi"], defasagem=defasagem)
        r = bt["retorno"].loc["2011":"2025"]
        taxas.append(m.retorno_anual(r))
    print(f"{ativo:10s} fechamento do sinal {taxas[0]:5.1%}   "
          f"um pregão de atraso {taxas[1]:5.1%}   "
          f"diferença {round(taxas[1] - taxas[0], 3) + 0.0:+.1%}")
