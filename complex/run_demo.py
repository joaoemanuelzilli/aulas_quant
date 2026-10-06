#!/usr/bin/env python3
"""
Script de demonstração rápida da estratégia.
Roda backtest completo e exibe resultados principais.
"""

import sys
import os

# Adicionar diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from backtest_engine import BacktestEngine
import numpy as np

def print_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def main():
    print_header("ADAPTIVE MULTI-FACTOR REGIME-SWITCHING STRATEGY")
    print("Demonstração de Backtest - 3 Anos de Dados Sintéticos")
    
    # Inicializar engine
    print("\n[1/4] Inicializando engine de backtest...")
    engine = BacktestEngine(initial_capital=1000000)
    
    # Gerar dados
    print("[2/4] Gerando dados sintéticos (756 dias, 50 ativos)...")
    returns_df, prices_df, fundamentals_df = engine.generate_synthetic_data(
        n_assets=50,
        n_days=756,
    )
    print(f"     ✓ Retornos: {returns_df.shape}")
    print(f"     ✓ Preços: {prices_df.shape}")
    print(f"     ✓ Fundamentos: {fundamentals_df.shape}")
    
    # Rodar backtest
    print("\n[3/4] Rodando backtest (pode levar 30-60 segundos)...")
    results = engine.run_backtest(
        returns_df, 
        prices_df, 
        fundamentals_df,
        rebalance_days=21
    )
    print(f"     ✓ Backtest completo: {len(results['dates'])} dias tradados")
    
    # Calcular métricas
    print("\n[4/4] Calculando métricas de performance...")
    metrics = engine.calculate_metrics(results)
    
    # Exibir relatório
    engine.print_report(metrics)
    
    # Exibir insights adicionais
    print_header("INSIGHTS ADICIONAIS")
    
    daily_returns = np.array(results['daily_returns'])
    
    print(f"\n📊 ESTATÍSTICAS DIÁRIAS:")
    print(f"   Retorno médio:        {np.mean(daily_returns):>8.3%}")
    print(f"   Retorno mediano:      {np.median(daily_returns):>8.3%}")
    print(f"   Desvio padrão:        {np.std(daily_returns):>8.3%}")
    print(f"   Skewness:             {np.mean(daily_returns**3) / np.std(daily_returns)**3:>8.3f}")
    print(f"   Kurtosis:             {np.mean(daily_returns**4) / np.std(daily_returns)**4:>8.3f}")
    
    # Analise de drawdown
    cumulative = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    
    print(f"\n📉 ANÁLISE DE DRAWDOWN:")
    print(f"   Máximo drawdown:      {np.min(drawdowns):>8.2%}")
    print(f"   Drawdown médio:       {np.mean(drawdowns):>8.2%}")
    print(f"   Dias em drawdown:     {np.sum(drawdowns < 0):>8} / {len(drawdowns)}")
    print(f"   Duração máxima DD:    {max_consecutive_negative(drawdowns):>8} dias")
    
    # Performance por trimestre
    print(f"\n📈 PERFORMANCE TRIMESTRAL:")
    quarterly_returns = []
    for i in range(0, len(daily_returns), 63):
        q_return = np.prod(1 + daily_returns[i:i+63]) - 1
        quarterly_returns.append(q_return)
    
    for i, qr in enumerate(quarterly_returns[:12]):  # Primeiros 3 anos
        print(f"   Q{(i % 4) + 1}/{i // 4 + 1}: {qr:>7.2%}")
    
    # Salvar arquivos
    import pandas as pd
    results_df = pd.DataFrame({
        'date': results['dates'],
        'portfolio_value': results['portfolio_value'],
        'daily_return': results['daily_returns'],
        'drawdown': results['drawdown'],
        'sharpe_rolling': results['sharpe_rolling'],
    })
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'backtest_results.csv'), index=False)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'strategy_metrics.csv'), index=False)
    
    print(f"\n✅ RESULTADOS SALVOS:")
    print(f"   outputs/backtest_results.csv (série temporal)")
    print(f"   outputs/strategy_metrics.csv (métricas consolidadas)")
    
    print_header("CONCLUSÃO")
    print(f"""
A estratégia demonstrou performance consistente:
  • Retorno anual de {metrics['annual_return']:.1%} 
  • Sharpe ratio de {metrics['sharpe_ratio']:.2f}
  • Máximo drawdown controlado em {metrics['max_drawdown']:.1%}
  • Taxa de acerto de {metrics['win_rate']:.1%}

Próximos passos:
  1. Compilar relatórios LaTeX (relatorio_*.tex → PDF)
  2. Revisar análise detalhada em relatorio_4_resultados.pdf
  3. Validar em dados reais antes de produção
    """)
    
    return metrics


def max_consecutive_negative(arr):
    """Encontra máxima sequência consecutiva de negativos."""
    max_consec = 0
    current_consec = 0
    for x in arr:
        if x < 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0
    return max_consec


if __name__ == "__main__":
    try:
        metrics = main()
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
