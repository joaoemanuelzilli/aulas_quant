"""
Aula 1 - Parte 3
Casos completos (mais próximos da vida real) para mostrar outputs.
"""

import numpy as np
import pandas as pd


def caso_1_resultado_corretora() -> None:
    """Caso real 1: análise de receita e resultado por mesa de operação."""
    print("=" * 70)
    print("CASO 1 - CORRETORA: RECEITA E RESULTADO POR MESA")
    print("=" * 70)

    # Dados fictícios de operações por mesa
    operacoes = pd.DataFrame(
        {
            "mesa": [
                "Ações",
                "Derivativos",
                "Ações",
                "Renda Fixa",
                "Derivativos",
                "Ações",
                "Renda Fixa",
            ],
            "receita_corretagem": [22000, 18500, 31000, 14000, 19500, 28000, 15500],
            "custo_operacional": [12000, 11000, 17000, 9000, 12500, 16000, 9800],
        }
    )

    # Calculando resultado e margem
    operacoes["resultado"] = operacoes["receita_corretagem"] - operacoes["custo_operacional"]
    operacoes["margem_pct"] = (operacoes["resultado"] / operacoes["receita_corretagem"]) * 100

    resumo = (
        operacoes.groupby("mesa", as_index=False)
        .agg(
            receita_total=("receita_corretagem", "sum"),
            resultado_total=("resultado", "sum"),
            margem_media=("margem_pct", "mean"),
        )
        .sort_values("receita_total", ascending=False)
    )

    print("Operações:")
    print(operacoes)
    print("\nResumo por mesa:")
    print(resumo.round(2))


def caso_2_anomalia_volume() -> None:
    """Caso real 2: detecção simples de anomalia em volume negociado."""
    print("\n" + "=" * 70)
    print("CASO 2 - MERCADO: DETECÇÃO DE ANOMALIAS EM VOLUME")
    print("=" * 70)

    # Série de volume diário negociado com pontos anômalos
    volumes = np.array([1.20, 1.18, 1.25, 1.22, 1.19, 2.40, 1.21, 1.17, 0.55, 1.23])

    # Regra simples: ponto fora de média +/- 2 desvios padrão
    media = np.mean(volumes)
    desvio = np.std(volumes)
    limite_inferior = media - 2 * desvio
    limite_superior = media + 2 * desvio

    anomalias = volumes[(volumes < limite_inferior) | (volumes > limite_superior)]

    print("Volumes (em milhões):", volumes)
    print("Média:", round(float(media), 2))
    print("Desvio padrão:", round(float(desvio), 2))
    print("Limites:", round(float(limite_inferior), 2), "a", round(float(limite_superior), 2))
    print("Anomalias encontradas:", anomalias)


def caso_3_carteira_investimentos() -> None:
    """Caso real 3: retorno médio e risco simples de uma carteira."""
    print("\n" + "=" * 70)
    print("CASO 3 - FINANÇAS: RETORNO E RISCO DE CARTEIRA")
    print("=" * 70)

    # Retornos mensais de 3 ativos
    retornos = pd.DataFrame(
        {
            "Ativo_A": [0.02, 0.01, -0.005, 0.018, 0.012, 0.007],
            "Ativo_B": [0.015, -0.01, 0.022, 0.01, 0.005, 0.011],
            "Ativo_C": [0.008, 0.012, 0.009, -0.004, 0.014, 0.01],
        }
    )

    # Pesos da carteira (soma = 1)
    pesos = np.array([0.5, 0.3, 0.2])

    # Retorno médio mensal por ativo
    retorno_medio_ativos = retornos.mean().values

    # Retorno esperado da carteira = soma(peso * retorno médio)
    retorno_esperado_carteira = np.dot(pesos, retorno_medio_ativos)

    # Risco simples: desvio padrão dos retornos mensais da carteira
    retorno_mensal_carteira = retornos.values @ pesos
    risco_carteira = np.std(retorno_mensal_carteira)

    print("Retornos mensais dos ativos:")
    print(retornos)
    print("\nPesos:", pesos)
    print("Retorno médio por ativo:", np.round(retorno_medio_ativos, 4))
    print("Retorno esperado da carteira:", round(float(retorno_esperado_carteira), 4))
    print("Risco (desvio padrão) da carteira:", round(float(risco_carteira), 4))


def main() -> None:
    # Executa os 3 casos em sequência
    caso_1_resultado_corretora()
    caso_2_anomalia_volume()
    caso_3_carteira_investimentos()


if __name__ == "__main__":
    main()
