"""
AULA 7 - Vieses em Backtesting
================================
TEMA 3: Overfitting (Sobreajuste)

O que é?
  A estratégia foi "ajustada demais" aos dados históricos.
  Ela aprendeu o ruído do passado em vez dos padrões reais.

Como acontece?
  - Testar centenas de combinações de parâmetros e escolher a melhor
  - Adicionar regras até o backtest ficar perfeito
  - Usar o mesmo período para otimizar E validar

Resultado:
  Backtest espetacular. Live trading: decepcionante.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import product

os.makedirs("aula7/outputs", exist_ok=True)

# --------------------------------------------------------------------------
# 1. DADOS
# --------------------------------------------------------------------------
df = yf.download("BBAS3.SA", start="2015-01-01", end="2024-01-01", auto_adjust=True)
close = df["Close"].squeeze()

# Dividir em treino (in-sample) e teste (out-of-sample)
split = int(len(close) * 0.70)
close_treino = close.iloc[:split]
close_teste  = close.iloc[split:]

print(f"Período de TREINO : {close_treino.index[0].date()} → {close_treino.index[-1].date()} ({len(close_treino)} dias)")
print(f"Período de TESTE  : {close_teste.index[0].date()}  → {close_teste.index[-1].date()}  ({len(close_teste)} dias)")
print()

# --------------------------------------------------------------------------
# 2. FUNÇÃO: backtest de cruzamento de médias
# --------------------------------------------------------------------------
def backtest_sma(precos, janela_rapida, janela_lenta):
    """Retorna o retorno total de uma estratégia de cruzamento de SMAs."""
    sma_rapida = precos.rolling(janela_rapida).mean()
    sma_lenta  = precos.rolling(janela_lenta).mean()

    sinal   = (sma_rapida > sma_lenta).astype(int)
    retorno = precos.pct_change()
    ret_estrategia = (sinal.shift(1) * retorno).dropna()

    return (1 + ret_estrategia).cumprod().iloc[-1]  # retorno acumulado final

# --------------------------------------------------------------------------
# 3. GRID SEARCH: encontrar os melhores parâmetros NO TREINO
# --------------------------------------------------------------------------
janelas_rapidas = range(5,  50,  5)
janelas_lentas  = range(20, 200, 10)

print("Otimizando parâmetros no período de TREINO...")
resultados = []

for jr, jl in product(janelas_rapidas, janelas_lentas):
    if jr >= jl:
        continue
    ret = backtest_sma(close_treino, jr, jl)
    resultados.append({"rapida": jr, "lenta": jl, "retorno_treino": ret})

df_res = pd.DataFrame(resultados).sort_values("retorno_treino", ascending=False)

melhor = df_res.iloc[0]
pior   = df_res.iloc[-1]

print(f"\nMelhor combinação no TREINO:  SMA({int(melhor['rapida'])}, {int(melhor['lenta'])})  → retorno {melhor['retorno_treino']:.2f}x")
print(f"Pior  combinação no TREINO:  SMA({int(pior['rapida'])}, {int(pior['lenta'])})   → retorno {pior['retorno_treino']:.2f}x")

# --------------------------------------------------------------------------
# 4. APLICAR NO PERÍODO DE TESTE (out-of-sample)
# --------------------------------------------------------------------------
ret_melhor_teste  = backtest_sma(close_teste, int(melhor["rapida"]), int(melhor["lenta"]))
ret_pior_teste    = backtest_sma(close_teste, int(pior["rapida"]),   int(pior["lenta"]))
ret_bh_teste      = close_teste.iloc[-1] / close_teste.iloc[0]  # Buy & Hold

print()
print("=" * 60)
print("RESULTADO NO PERÍODO DE TESTE (fora da amostra)")
print("=" * 60)
print(f"'Melhor' estratégia (treino):  {ret_melhor_teste:.2f}x")
print(f"'Pior'   estratégia (treino):  {ret_pior_teste:.2f}x")
print(f"Buy & Hold (benchmark):        {ret_bh_teste:.2f}x")
print()
print("Observe que a 'melhor' estratégia do treino raramente")
print("mantém a liderança no período de teste — é o overfitting!\n")

# --------------------------------------------------------------------------
# 5. VISUALIZAÇÃO: distribuição dos retornos no treino vs. teste
# --------------------------------------------------------------------------
retornos_treino = df_res["retorno_treino"].values

# Calcular retorno no teste para todas as combinações
df_res["retorno_teste"] = df_res.apply(
    lambda r: backtest_sma(close_teste, int(r["rapida"]), int(r["lenta"])), axis=1
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_res["retorno_treino"], bins=30, color="steelblue", edgecolor="white")
axes[0].axvline(melhor["retorno_treino"], color="red", linestyle="--", label=f"Melhor treino: {melhor['retorno_treino']:.2f}x")
axes[0].set_title("Distribuição de retornos — TREINO (in-sample)")
axes[0].set_xlabel("Retorno acumulado")
axes[0].legend()

axes[1].hist(df_res["retorno_teste"], bins=30, color="salmon", edgecolor="white")
axes[1].axvline(ret_melhor_teste, color="red",  linestyle="--", label=f"'Melhor' no teste: {ret_melhor_teste:.2f}x")
axes[1].axvline(ret_bh_teste,     color="green", linestyle="--", label=f"Buy & Hold: {ret_bh_teste:.2f}x")
axes[1].set_title("Distribuição de retornos — TESTE (out-of-sample)")
axes[1].set_xlabel("Retorno acumulado")
axes[1].legend()

plt.suptitle("Overfitting: a 'melhor' estratégia raramente transfere para fora da amostra")
plt.tight_layout()
plt.savefig("aula7/outputs/03_overfitting.png", dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 6. COMO EVITAR
# --------------------------------------------------------------------------
print("=" * 60)
print("COMO EVITAR O OVERFITTING")
print("=" * 60)
print("""
1. Separar dados em treino / validação / teste:
   - Otimize nos dados de treino
   - Ajuste final apenas uma vez no teste (nunca re-otimize no teste!)

2. Walk-forward optimization:
   - Otimiza em janela rolling e valida na janela seguinte
   - Mais realista que um split único

3. Manter o número de parâmetros pequeno:
   - Estratégias com 2-3 parâmetros são mais robustas que 10+

4. Teste de robustez de parâmetros:
   - Se só funciona com SMA(47, 183) e não com (45, 180), desconfie

5. Out-of-sample obrigatório:
   - Nunca reportar resultado sem período de teste que NUNCA foi tocado
""")
