"""
AULA 7 - Vieses em Backtesting
================================
TEMA 1: Look-Ahead Bias (Viés de Antecipação)

O que é?
  Quando a estratégia usa informações que NÃO estariam disponíveis
  no momento da decisão de trading. O modelo "vê o futuro".

Exemplos comuns:
  - Usar a máxima/mínima do dia para decidir a entrada naquele mesmo dia
  - Calcular média sobre uma janela que inclui dados futuros
  - Normalizar com estatísticas calculadas sobre TODO o histórico
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

os.makedirs("aula7/outputs", exist_ok=True)

# --------------------------------------------------------------------------
# 1. DADOS
# --------------------------------------------------------------------------
df = yf.download("PETR4.SA", start="2020-01-01", end="2024-01-01", auto_adjust=True)
close = df["Close"].squeeze()
print(f"Dados carregados: {len(close)} pregões\n")

# --------------------------------------------------------------------------
# 2. DEMONSTRAÇÃO: média móvel COM look-ahead bias
# --------------------------------------------------------------------------
# centered=True calcula a média usando N/2 pontos ANTES e N/2 DEPOIS do ponto atual.
# Em produção isso é impossível: você não sabe os preços futuros!
sma_lookahead = close.rolling(window=20, center=True).mean()

# --------------------------------------------------------------------------
# 3. CORRETO: média móvel SEM look-ahead bias
# --------------------------------------------------------------------------
# center=False (padrão) usa apenas os últimos 20 preços — só o passado.
sma_correto = close.rolling(window=20).mean()

# --------------------------------------------------------------------------
# 4. VISUALIZAÇÃO
# --------------------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(close, label="Preço", color="gray", alpha=0.6, linewidth=1)
plt.plot(sma_lookahead, label="SMA20 COM look-ahead (center=True)", color="red", linewidth=2)
plt.plot(sma_correto,   label="SMA20 SEM look-ahead (correto)",     color="green", linewidth=2)
plt.title("Look-Ahead Bias: SMA centrada vs. SMA correta")
plt.legend()
plt.tight_layout()
plt.savefig("aula7/outputs/01_sma_lookahead.png", dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 5. IMPACTO NO SINAL DE TRADING
# --------------------------------------------------------------------------
print("=" * 60)
print("IMPACTO NO SINAL DE TRADING")
print("=" * 60)

# Sinal com viés: compra quando preço > SMA centrada (usa futuro)
sinal_bias    = (close > sma_lookahead).astype(int)

# Sinal correto: compra quando preço > SMA passada
sinal_correto = (close > sma_correto).astype(int)

# Retorno diário
retorno_diario = close.pct_change()

# Retorno da estratégia (posição do dia anterior aplicada ao dia atual)
ret_bias    = (sinal_bias.shift(1)    * retorno_diario).dropna()
ret_correto = (sinal_correto.shift(1) * retorno_diario).dropna()

retorno_acum_bias    = (1 + ret_bias).cumprod()
retorno_acum_correto = (1 + ret_correto).cumprod()

print(f"Retorno acumulado COM look-ahead bias : {retorno_acum_bias.iloc[-1]:.2f}x")
print(f"Retorno acumulado SEM look-ahead bias : {retorno_acum_correto.iloc[-1]:.2f}x")
print()
print("Nota: a estratégia com viés parece MUITO melhor — mas é uma ilusão!")
print("Ela 'sabe' onde o preço vai, o que é impossível em trading real.\n")

plt.figure(figsize=(12, 4))
retorno_acum_bias.plot(label="COM look-ahead bias (irreal)", color="red")
retorno_acum_correto.plot(label="SEM look-ahead bias (real)", color="green")
plt.title("Curva de capital: COM vs. SEM look-ahead bias")
plt.ylabel("Retorno acumulado")
plt.legend()
plt.tight_layout()
plt.savefig("aula7/outputs/02_capital_curve.png", dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 6. OUTRO EXEMPLO CLÁSSICO: normalização com dados futuros
# --------------------------------------------------------------------------
print("=" * 60)
print("OUTRO EXEMPLO: Normalização incorreta")
print("=" * 60)

# ERRADO: normaliza com min/max de TODA a série (inclui futuro no treino)
norm_errado = (close - close.min()) / (close.max() - close.min())

# CORRETO: normaliza de forma expansiva (só dados passados disponíveis)
norm_correto = (close - close.expanding().min()) / (
    close.expanding().max() - close.expanding().min()
)

print("Normalização ERRADA usa o mínimo e máximo de toda a série.")
print("Isso significa que no dia 1 você já 'sabia' o preço máximo dos próximos 4 anos.")
print("Normalização CORRETA usa apenas min/max histórico até aquele momento.\n")
