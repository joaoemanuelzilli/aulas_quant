import pandas as pd

BRUTO = "../../dados/bruto/"
precos = pd.read_csv(BRUTO + "precos_ajustados.csv",
                     index_col=0, parse_dates=True)
oficial = pd.read_csv(BRUTO + "b3_cotahist.csv",
                      parse_dates=["data"])
ativos = ["PETR4.SA", "MGLU3.SA", "^BVSP", "^GSPC", "BRL=X"]
p = precos[ativos]

diagnostico = pd.DataFrame({
    "primeira data": p.apply(pd.Series.first_valid_index),
    "faltantes": p.isna().sum(),
    "% do total": (p.isna().mean() * 100).round(1),
})
print(diagnostico)

# quem estava aberto em cada data? O calendário da B3 vem do
# arquivo oficial; o da NYSE, de onde o S&P 500 tem valor
b3 = p.index.isin(oficial["data"])
nyse = p["^GSPC"].notna()
print(f"\nlinhas no arquivo:        {len(p)}")
print(f"B3 e NYSE abertas:        {(b3 & nyse).sum()}")
print(f"só a NYSE aberta:         {(~b3 & nyse).sum()}")
print(f"só a B3 aberta:           {(b3 & ~nyse).sum()}")
print(f"nenhuma aberta:           {(~b3 & ~nyse).sum()}")


def datas(filtro):
    return [f"{d:%d/%m/%Y}" for d in p.index[filtro]]


tem = p["PETR4.SA"].notna()
print(f"\nPETR4 com preço em dia sem pregão: {(tem & ~b3).sum()}")
print("pregão sem PETR4:", datas(b3 & ~tem))
print("pregão sem Ibovespa:", datas(b3 & p["^BVSP"].isna()))
