"""
BACKTEST DO MODELO: ADAPTIVE MULTI-FACTOR REGIME-SWITCHING
===========================================================

Realiza backtest completo da estratégia com rebalanceamento dinâmico,
controle de riscos e análise detalhada de performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Importar modelo
from thesis_and_model import AdaptiveMultiFactorModel, StrategyMetrics


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class BacktestEngine:
    """
    Engine de backtest com rebalanceamento dinâmico.
    """
    
    def __init__(self, initial_capital=1000000, rebalance_freq='monthly'):
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.portfolio_values = [initial_capital]
        self.positions = []
        self.returns = []
        self.metrics_over_time = []
        
    def generate_synthetic_data(self, n_assets=50, n_days=756, n_fundamentals=6):
        """
        Gera dados sintéticos realistas para teste.
        """
        np.random.seed(42)
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Retornos com componentes de mean-reversion e momentum
        returns_data = np.random.randn(n_days, n_assets) * 0.015
        
        # Adicionar componentes de tendência e mean-reversion
        for i in range(n_assets):
            trend = np.sin(np.linspace(0, 4*np.pi, n_days)) * 0.005
            returns_data[:, i] += trend
            
            # Mean-reversion: reverter para a média
            cumulative = np.cumprod(1 + returns_data[:, i])
            mean_price = cumulative.mean()
            mean_reversion = -0.0001 * (cumulative - mean_price) / mean_price
            returns_data[:, i] += mean_reversion
        
        returns_df = pd.DataFrame(returns_data, index=dates, 
                                 columns=[f'Asset_{i}' for i in range(n_assets)])
        
        # Preços (começando com 100)
        prices_df = pd.DataFrame(
            100 * np.cumprod(1 + returns_data, axis=0),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Dados fundamentalistas (simulados)
        fundamentals = pd.DataFrame({
            'book_value': np.random.uniform(10, 100, n_assets),
            'roe': np.random.uniform(0.05, 0.25, n_assets),
            'profit_margin': np.random.uniform(0.01, 0.20, n_assets),
            'revenue_growth': np.random.uniform(-0.05, 0.20, n_assets),
        }, index=returns_df.columns)
        
        return returns_df, prices_df, fundamentals
    
    def run_backtest(self, returns_df, prices_df, fundamentals_df, 
                     rebalance_days=63):
        """
        Executa backtest completo.
        """
        model = AdaptiveMultiFactorModel()
        portfolio_value = self.initial_capital
        positions = np.zeros(len(returns_df.columns))
        
        results = {
            'dates': [],
            'portfolio_value': [],
            'daily_returns': [],
            'positions': [],
            'regime': [],
            'predicted_returns': [],
            'sharpe_rolling': [],
            'drawdown': []
        }
        
        max_value = self.initial_capital
        
        # Backtest loop
        for day in range(252, len(returns_df)):  # Começar após um ano de dados
            current_date = returns_df.index[day]
            
            # Rebalanceamento a cada N dias
            if (day - 252) % rebalance_days == 0 and day > 252:
                # Usar dados históricos até o dia anterior
                hist_returns = returns_df.iloc[:day]
                hist_prices = prices_df.iloc[:day]
                
                # Gerar sinais (usando método refatorado AFML-compliant)
                signals = model.generate_robust_signals(
                    hist_returns.iloc[-252:],
                    hist_prices.iloc[-252:],
                    fundamentals_df
                )
                
                weights = signals['weights']
                
                # Converter pesos para posições
                positions = (weights * portfolio_value) / (prices_df.iloc[day].values + 1e-8)
            
            # Computar retorno diário
            daily_return = (returns_df.iloc[day].values * positions).sum() / portfolio_value
            daily_return = np.clip(daily_return, -0.10, 0.10)  # Limite de 10% por dia
            
            portfolio_value *= (1 + daily_return)
            
            # Armazenar métricas
            results['dates'].append(current_date)
            results['portfolio_value'].append(portfolio_value)
            results['daily_returns'].append(daily_return)
            results['positions'].append(positions.copy())
            
            # Calcular drawdown
            max_value = max(max_value, portfolio_value)
            drawdown = (portfolio_value - max_value) / max_value
            results['drawdown'].append(drawdown)
            
            # Sharpe rolling (últimos 63 dias)
            if len(results['daily_returns']) >= 63:
                recent_returns = np.array(results['daily_returns'][-63:])
                sharpe = StrategyMetrics.sharpe_ratio(recent_returns)
                results['sharpe_rolling'].append(sharpe)
            else:
                results['sharpe_rolling'].append(np.nan)
        
        return results
    
    def calculate_metrics(self, results):
        """
        Calcula métricas detalhadas de performance.
        """
        daily_returns = np.array(results['daily_returns'])
        portfolio_values = np.array(results['portfolio_value'])
        
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = (portfolio_values[-1] / self.initial_capital) ** (252 / len(daily_returns)) - 1
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': StrategyMetrics.sharpe_ratio(daily_returns),
            'sortino_ratio': StrategyMetrics.sortino_ratio(daily_returns),
            'max_drawdown': StrategyMetrics.max_drawdown(daily_returns),
            'calmar_ratio': StrategyMetrics.calmar_ratio(
                daily_returns, 
                StrategyMetrics.max_drawdown(daily_returns)
            ),
            'var_95': StrategyMetrics.var_95(daily_returns),
            'cvar_95': StrategyMetrics.cvar_95(daily_returns),
            'win_rate': StrategyMetrics.win_rate(daily_returns),
            'profit_factor': StrategyMetrics.profit_factor(daily_returns),
            'volatility': np.std(daily_returns) * np.sqrt(252),
            'n_trades': np.sum(np.abs(np.diff(np.array(results['positions']), axis=0)) > 1e-6),
            'average_trade_return': np.mean(daily_returns[daily_returns != 0]) if np.any(daily_returns != 0) else 0,
            'positive_days': np.sum(daily_returns > 0),
            'total_days': len(daily_returns),
            'best_day': np.max(daily_returns),
            'worst_day': np.min(daily_returns),
        }
        
        return metrics
    
    def print_report(self, metrics):
        """
        Imprime relatório formatado de performance.
        """
        print("\n" + "="*70)
        print("BACKTEST RESULTS - ADAPTIVE MULTI-FACTOR REGIME-SWITCHING STRATEGY")
        print("="*70)
        
        print(f"\nRETURNS:")
        print(f"  Total Return:          {metrics['total_return']:>10.2%}")
        print(f"  Annual Return:         {metrics['annual_return']:>10.2%}")
        print(f"  Volatility:            {metrics['volatility']:>10.2%}")
        
        print(f"\nRISK-ADJUSTED METRICS:")
        print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:         {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:          {metrics['calmar_ratio']:>10.2f}")
        print(f"  Max Drawdown:          {metrics['max_drawdown']:>10.2%}")
        
        print(f"\nRISK METRICS:")
        print(f"  VaR (95%):             {metrics['var_95']:>10.2%}")
        print(f"  CVaR (95%):            {metrics['cvar_95']:>10.2%}")
        
        print(f"\nTRADE STATISTICS:")
        print(f"  Win Rate:              {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:         {metrics['profit_factor']:>10.2f}")
        print(f"  Number of Trades:      {int(metrics['n_trades']):>10}")
        print(f"  Avg Trade Return:      {metrics['average_trade_return']:>10.2%}")
        
        print(f"\nEXTREMES:")
        print(f"  Best Day:              {metrics['best_day']:>10.2%}")
        print(f"  Worst Day:             {metrics['worst_day']:>10.2%}")
        print(f"  Positive Days:         {metrics['positive_days']:>10} / {metrics['total_days']}")
        
        print("="*70 + "\n")
        
        return metrics


def main():
    print("Iniciando backtest...")
    
    # Gerar dados sintéticos
    engine = BacktestEngine(initial_capital=1000000)
    returns_df, prices_df, fundamentals_df = engine.generate_synthetic_data(
        n_assets=50,
        n_days=756,  # 3 anos
    )
    
    # Rodar backtest
    results = engine.run_backtest(
        returns_df, 
        prices_df, 
        fundamentals_df,
        rebalance_days=21  # Rebalancear mensalmente
    )
    
    # Calcular métricas
    metrics = engine.calculate_metrics(results)
    
    # Salvar resultados
    results_df = pd.DataFrame({
        'date': results['dates'],
        'portfolio_value': results['portfolio_value'],
        'daily_return': results['daily_returns'],
        'drawdown': results['drawdown'],
        'sharpe_rolling': results['sharpe_rolling'],
    })
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'backtest_results.csv'), index=False)
    
    # Imprimir relatório
    engine.print_report(metrics)
    
    # Salvar métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'strategy_metrics.csv'), index=False)
    
    return results_df, metrics


if __name__ == "__main__":
    results_df, metrics = main()
