"""Ajuste diário de um contrato de WTI (junho de 2020)."""
import pandas as pd
from futuros import ler, mes_do_contrato

d = ler(["cl1", "cl2"], "2020-04-14", "2020-04-24")
uteis = ler(["cl1"]).index
junho = pd.Period("2020-06")          # mês de entrega carregado
# em cada data, o contrato de junho é o 1 ou o 2 da EIA
k = {t: 1 if mes_do_contrato(t, 1, uteis, "cl") == junho else 2
     for t in d.index}
preco = pd.Series({t: d.at[t, f"cl{k[t]}"] for t in d.index})
ajuste = preco.diff() * 1000          # 1.000 barris por contrato
conta = 10_000 + ajuste.fillna(0).cumsum()   # depósito inicial
print(pd.DataFrame({"preço": preco, "ajuste": ajuste,
                    "conta": conta}).round(2).to_string())
variacao = (preco.iloc[-1] - preco.iloc[0]) * 1000
assert abs(ajuste.sum() - variacao) < 1e-6
