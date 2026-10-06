"""
AULA 7 - Vieses em Backtesting
================================
TEMA 2: Survivorship Bias (Viés de Sobrevivência)

O que é?
  Testar a estratégia APENAS nas ações que ainda existem hoje.
  As empresas que faliram, foram deslistadas ou incorporadas são ignoradas.

Resultado:
  O backtest parece muito melhor do que seria na realidade,
  pois você evita naturalmente as piores perdas.

Exemplo real:
  Se você pegar as 500 maiores empresas do S&P500 HOJE e testar
  uma estratégia desde 2000, ignora todas as empresas que estavam
  no índice em 2000 mas faliram (Enron, Lehman Brothers, etc.).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("aula7/outputs", exist_ok=True)

np.random.seed(42)

# --------------------------------------------------------------------------
# 1. SIMULAÇÃO: universo de 100 empresas
# --------------------------------------------------------------------------
N_EMPRESAS   = 100
N_DIAS       = 252 * 5  # 5 anos
PROB_FALECIA = 0.003     # 0.3% de chance de falir por dia (realista)

print("=" * 60)
print("SIMULAÇÃO: 100 empresas ao longo de 5 anos")
print("=" * 60)

retornos_todas    = {}
retornos_vivas    = {}
empresas_falidas  = []

for i in range(N_EMPRESAS):
    nome = f"EMP_{i:03d}"
    precos = [100.0]
    faliu  = False

    for dia in range(N_DIAS):
        if np.random.rand() < PROB_FALECIA:
            # Empresa vai à falência: perde quase tudo de repente
            precos.append(precos[-1] * np.random.uniform(0.01, 0.10))
            faliu = True
            break
        else:
            retorno_diario = np.random.normal(0.0003, 0.02)
            precos.append(precos[-1] * (1 + retorno_diario))

    serie = pd.Series(precos)
    retornos_todas[nome] = serie.pct_change().dropna()

    if not faliu:
        retornos_vivas[nome] = serie.pct_change().dropna()
    else:
        empresas_falidas.append(nome)

print(f"Empresas que sobreviveram: {len(retornos_vivas)}")
print(f"Empresas que faliram:      {len(empresas_falidas)}")
print()

# --------------------------------------------------------------------------
# 2. ESTRATÉGIA SIMPLES: comprar e segurar (buy & hold) carteira igualmente ponderada
# --------------------------------------------------------------------------

def retorno_carteira(dict_retornos):
    """Calcula retorno acumulado de carteira igualmente ponderada."""
    df = pd.DataFrame(dict_retornos)
    # Preenche com 0 após falência (empresa deixou de existir)
    df = df.fillna(0)
    ret_carteira = df.mean(axis=1)
    return (1 + ret_carteira).cumprod()

# Backtest ERRADO: só empresas vivas hoje (survivorship bias)
curva_so_vivas = retorno_carteira(retornos_vivas)

# Backtest CORRETO: todas as empresas (incluindo as que faliram)
curva_todas = retorno_carteira(retornos_todas)

print("=" * 60)
print("RESULTADO DO BACKTEST")
print("=" * 60)
print(f"Retorno final SÓ com sobreviventes (com viés):  {curva_so_vivas.iloc[-1]:.2f}x")
print(f"Retorno final com TODAS as empresas (correto):  {curva_todas.iloc[-1]:.2f}x")
print()
print("Diferença causada pelo survivorship bias:")
print(f"  {((curva_so_vivas.iloc[-1] / curva_todas.iloc[-1]) - 1) * 100:.1f}% de resultado inflado!\n")

# --------------------------------------------------------------------------
# 3. VISUALIZAÇÃO
# --------------------------------------------------------------------------
plt.figure(figsize=(12, 5))
curva_so_vivas.plot(label=f"Só sobreviventes ({len(retornos_vivas)} empresas)", color="red",   linewidth=2)
curva_todas.plot(   label=f"Universo completo ({N_EMPRESAS} empresas)",          color="green", linewidth=2)
plt.axhline(1, linestyle="--", color="gray", linewidth=1)
plt.title("Survivorship Bias: impacto no retorno da carteira")
plt.ylabel("Retorno acumulado (R$1 investido)")
plt.legend()
plt.tight_layout()
plt.savefig("aula7/outputs/02_survivorship_bias.png", dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 4. COMO EVITAR
# --------------------------------------------------------------------------
print("=" * 60)
print("COMO EVITAR O SURVIVORSHIP BIAS")
print("=" * 60)
print("""
1. Use bases de dados point-in-time:
   - Registram quais ativos existiam em cada data
   - Ex: Refinitiv, Bloomberg, CRSP (mercado americano)

2. Para o mercado brasileiro:
   - B3 disponibiliza histórico de carteiras de índices (IBr-X, Small Caps)
   - Guardar snapshots periódicos dos componentes do índice

3. Inclua ativos deslistados no seu universo de teste:
   - Ações canceladas por falência, incorporação, saída de bolsa

4. Cuidado ao usar screeners do tipo "Top 10 do Ibovespa":
   - O índice de hoje não é o mesmo de 10 anos atrás
""")
