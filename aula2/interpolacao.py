import pandas as pd
import numpy as np

dados = {
    "Nome": ["Ana", "Bruno", "Carlos", "Daniela", "Eduardo"],
    "Idade": [23, np.nan, 35, np.nan, 29],
    "Salario": [2500, 3200, np.nan, 4100, np.nan]
}



df = pd.DataFrame(dados)

print("=== DataFrame original ===")
print(df)

# -------------------------------
# 1. Identificar dados faltantes
# -------------------------------
print("\n=== Valores faltantes por coluna ===")
print(df.isnull().sum())

# -------------------------------
# 2. Remover linhas com NaN
# -------------------------------
df_drop = df.dropna()
print("\n=== DataFrame sem valores faltantes (dropna) ===")
print(df_drop)

# -------------------------------
# 3. Preencher com média
# -------------------------------
df_media = df.copy()
df_media["Idade"].fillna(df_media["Idade"].mean(), inplace=True)
df_media["Salario"].fillna(df_media["Salario"].mean(), inplace=True)

print("\n=== Preenchido com média ===")
print(df_media)

# -------------------------------
# 4. Preencher com valor fixo
# -------------------------------
df_fixo = df.fillna(0)
print("\n=== Preenchido com 0 ===")
print(df_fixo)

# -------------------------------
# 5. Forward Fill (propagar valor anterior)
# -------------------------------++
df_ffill = df.fillna(method="ffill")
print("\n=== Forward Fill ===")
print(df_ffill)

# -------------------------------
# 6. Interpolação c fx
# -------------------------------
df_interp = df.copy()
df_interp["Idade"] = df_interp["Idade"].interpolate()
df_interp["Salario"] = df_interp["Salario"].interpolate()

print("\n=== Interpolação ===")
print(df_interp)