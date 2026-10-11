import pandas as pd


def ler(nome):
    return pd.read_csv("../../dados/" + nome, index_col=0,
                       parse_dates=True)


oficial = pd.read_csv("../../dados/bruto/b3_cotahist.csv",
                      parse_dates=["data"]).pivot(
    index="data", columns="codigo", values="fechamento")
sem, ajustado = ler("precos_sem_ajuste.csv"), ler(
    "precos_ajustados.csv")

# desdobramento de 1 para 2 da WEG, data ex em 28/04/2021
dias = slice("2021-04-27", "2021-04-28")
weg = pd.DataFrame({"oficial": oficial.loc[dias, "WEGE3"],
                    "sem ajuste": sem.loc[dias, "WEGE3.SA"]})
weg["fator"] = weg["sem ajuste"] / weg["oficial"]
print(weg.round(3))
for nome in ["oficial", "sem ajuste"]:
    r = weg[nome].iloc[1] / weg[nome].iloc[0] - 1
    print(f"retorno pelo preço {nome}: {r:+.1%}")

# provento de R$ 3,3489 de PETR4, data ex em 22/11/2022
dias = slice("2022-11-21", "2022-11-22")
petr = pd.DataFrame({"sem ajuste": sem.loc[dias, "PETR4.SA"],
                     "ajustado": ajustado.loc[dias, "PETR4.SA"]})
petr["fator"] = petr["ajustado"] / petr["sem ajuste"]
print("\n", petr.round(4), sep="")
degrau = petr["fator"].iloc[0] / petr["fator"].iloc[1]
print(f"degrau do fator: {degrau:.4f}   "
      f"1 - 3,3489/26,78 = {1 - 3.3489 / 26.78:.4f}")
for nome in ["sem ajuste", "ajustado"]:
    r = petr[nome].iloc[1] / petr[nome].iloc[0] - 1
    print(f"retorno pelo preço {nome}: {r:+.2%}")

# a base pronta: um preço por ação em cada pregão da B3
acoes = [c for c in ajustado.columns if c.endswith(".SA")]
pregoes = ajustado.index.isin(oficial.index)
faltam = ajustado.loc[pregoes, acoes].isna().sum()
print(f"\npregões: {pregoes.sum()}   preços ausentes em pregão: "
      f"{faltam.drop('MGLU3.SA').sum()}")
print(f"MGLU3 antes da estreia: {faltam['MGLU3.SA']} pregões")
r = ajustado[acoes].pct_change(fill_method=None)
for papel, dia in [("PETR4.SA", "2011-08-17"),
                   ("ITUB4.SA", "2021-10-04"),
                   ("PETR4.SA", "2024-03-08")]:
    print(f"{papel[:5]} em {dia}: {r.loc[dia, papel]:+.2%}")
