"""
Aula 1 - Parte 2
Introdução ao pandas com dados tabulares.
"""

import pandas as pd
from pathlib import Path


print("=" * 70)
print("1) LEITURA DE DADOS")
print("=" * 70)

# Lendo o CSV para um DataFrame
base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / "dados" / "dados.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()

# Convertendo a coluna de data para tipo datetime
df["data"] = pd.to_datetime(df["data"])

print("Primeiras linhas:")
print(df.head())


print("\n" + "=" * 70)
print("2) EXPLORAÇÃO RÁPIDA")
print("=" * 70)

# Informações gerais da tabela
print("Total de linhas:", len(df))
print("Colunas:", list(df.columns))

# Criando coluna de faturamento por linha
df["faturamento"] = df["quantidade"] * df["preco_unitario"]

print("Faturamento total:", float(df["faturamento"].sum()))
print("Ticket médio por venda:", round(float(df["faturamento"].mean()), 2))


print("\n" + "=" * 70)
print("3) FILTROS")
print("=" * 70)

# Exemplo: vendas apenas de São Paulo
df_sp = df[df["cidade"] == "Sao Paulo"]
print("Vendas em São Paulo:")
print(df_sp[["data", "produto", "quantidade", "faturamento"]].head())

# Exemplo: produtos da categoria Gamer com quantidade >= 3
filtro_gamer = df[(df["categoria"] == "Gamer") & (df["quantidade"] >= 3)]
print("\nVendas Gamer com quantidade >= 3:")
print(filtro_gamer[["produto", "cidade", "quantidade", "faturamento"]])


print("\n" + "=" * 70)
print("4) AGRUPAMENTOS (GROUPBY)")
print("=" * 70)

# Faturamento total por cidade
fat_por_cidade = (
    df.groupby("cidade", as_index=False)["faturamento"]
    .sum()
    .sort_values("faturamento", ascending=False)
)
print("Faturamento por cidade:")
print(fat_por_cidade)

# Quantidade vendida por categoria
qtd_por_categoria = (
    df.groupby("categoria", as_index=False)["quantidade"]
    .sum()
    .sort_values("quantidade", ascending=False)
)
print("\nQuantidade por categoria:")
print(qtd_por_categoria)


print("\n" + "=" * 70)
print("5) JUNÇÃO (MERGE) COM META")
print("=" * 70)

# Tabela de metas fictícias por cidade
metas = pd.DataFrame(
    {
        "cidade": ["Sao Paulo", "Rio de Janeiro", "Curitiba", "Belo Horizonte"],
        "meta_faturamento": [15000, 7000, 6000, 5000],
    }
)

# Juntando o faturamento com as metas
comparativo = fat_por_cidade.merge(metas, on="cidade", how="left")
comparativo["atingiu_meta"] = comparativo["faturamento"] >= comparativo["meta_faturamento"]

print("Comparativo faturamento x meta:")
print(comparativo)


print("\nFim da parte de pandas ✅")