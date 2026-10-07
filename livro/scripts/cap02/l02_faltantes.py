import pandas as pd

BRUTO = "../../dados/bruto/"
precos = pd.read_csv(BRUTO + "precos_ajustados.csv",
                     index_col=0, parse_dates=True)
volume = pd.read_csv(BRUTO + "volume.csv",
                     index_col=0, parse_dates=True)
ativos = ["PETR4.SA", "MGLU3.SA", "^BVSP", "^GSPC", "BRL=X"]
p = precos[ativos]

diagnostico = pd.DataFrame({
    "primeira data": p.apply(pd.Series.first_valid_index),
    "faltantes": p.isna().sum(),
    "% do total": (p.isna().mean() * 100).round(1),
})
print(diagnostico)

# quem estava aberto em cada data? Houve pregão na B3 se
# alguma ação teve negócio; na NYSE, se o S&P 500 tem valor
acoes = volume.filter(like=".SA")     # só as ações
b3 = (acoes > 0).any(axis=1).reindex(p.index)
nyse = p["^GSPC"].notna()
print(f"\nlinhas no arquivo:        {len(p)}")
print(f"B3 e NYSE abertas:        {(b3 & nyse).sum()}")
print(f"só a NYSE aberta:         {(~b3 & nyse).sum()}")
print(f"só a B3 aberta:           {(b3 & ~nyse).sum()}")
print(f"nenhuma aberta:           {(~b3 & ~nyse).sum()}")

sem_pregao = ~b3 & p["PETR4.SA"].notna()
print(f"\nPETR4 com preço em dia sem pregão: {sem_pregao.sum()}")
sem_indice = b3 & p["^BVSP"].isna()
print("pregão sem Ibovespa:",
      [f"{d:%d/%m/%Y}" for d in p.index[sem_indice]])
