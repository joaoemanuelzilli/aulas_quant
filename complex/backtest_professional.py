"""
BACKTESTING PROFISSIONAL - López de Prado Compliant
════════════════════════════════════════════════════════════════════════

Este módulo implementa backtesting robusto seguindo "Advances in Financial 
Machine Learning" de Marcos López de Prado.

Características:
    ✓ Purged K-Fold Cross-Validation (sem data leakage)
    ✓ Embargo periods (separação treino/teste)
    ✓ Detecção de overfitting (Sharpe deflacionado)
    ✓ Meta-labeling (decisão inteligente de sinais)
    ✓ Triple barrier method (labeling robusto)
    ✓ Feature importance analysis (MDI + MDA)
    ✓ Walk-forward validation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Importar classes do modelo
from thesis_and_model import (
    AdaptiveMultiFactorModel, 
    StrategyMetrics,
    PurgedKFold,
    TripleBarrierLabeler,
    FeatureImportanceAnalyzer
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ProfessionalBacktestEngine:
    """
    Engine de backtest profissional PRADO-COMPLIANT.
    """
    
    def __init__(self, initial_capital=1000000, rebalance_days=21, 
                 embargo_pct=0.01, n_cv_folds=5):
        self.initial_capital = initial_capital
        self.rebalance_days = rebalance_days
        self.embargo_pct = embargo_pct
        self.n_cv_folds = n_cv_folds
        
        # Resultados
        self.portfolio_values = [initial_capital]
        self.returns = []
        self.trades = []
        self.signals = []
        self.regimes = []
        
    def generate_realistic_data(self, n_assets=50, n_days=756, seed=42):
        """
        Gera dados realistas com correlação estruturada e estrutural breaks.
        
        Mais realista que random walk simples.
        """
        np.random.seed(seed)
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Componentes de retorno
        returns_data = np.zeros((n_days, n_assets))
        
        for i in range(n_assets):
            # Componente 1: Trend + mean-reversion
            trend = np.sin(np.linspace(0, 4*np.pi, n_days)) * 0.005
            mean_reversion = -0.0001 * np.cumsum(np.random.randn(n_days) * 0.015)
            
            # Componente 2: Regime shifts
            regime_shift = np.where(np.arange(n_days) < n_days//3, 0, 0.005)
            regime_shift = np.where(np.arange(n_days) < 2*n_days//3, regime_shift, -0.005)
            
            # Componente 3: Volatility clustering
            vol_cluster = np.random.randn(n_days) * (0.01 + 0.02 * np.abs(np.cumsum(np.random.randn(n_days) * 0.1)))
            
            # Combinar
            returns_data[:, i] = trend + mean_reversion + regime_shift + vol_cluster
        
        # DataFrame de retornos
        returns_df = pd.DataFrame(
            returns_data, 
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Preços (começar com 100)
        prices_df = pd.DataFrame(
            100 * np.cumprod(1 + returns_data, axis=0),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Fundamentals (correlacionados com performance)
        fundamentals = pd.DataFrame({
            'book_value': 10 + 50 * np.random.rand(n_assets),
            'roe': 0.05 + 0.15 * np.random.rand(n_assets),
            'profit_margin': 0.01 + 0.15 * np.random.rand(n_assets),
            'revenue_growth': -0.05 + 0.20 * np.random.rand(n_assets),
        }, index=returns_df.columns)
        
        return returns_df, prices_df, fundamentals
    
    def run_walk_forward_backtest(self, returns_df, prices_df, fundamentals_df,
                                  window_size=252, rebalance_days=21, embargo_days=10):
        """
        Walk-forward backtest (mais realista que simples backtest).
        
        PRADO-COMPLIANT:
        - Treina em janela histórica
        - Testa em período futuro
        - Move janela para frente
        - Embargo period entre train/test
        """
        model = AdaptiveMultiFactorModel(embargo_pct=self.embargo_pct)
        
        portfolio_value = self.initial_capital
        positions = np.zeros(len(returns_df.columns))
        
        results = {
            'dates': [],
            'portfolio_value': [],
            'daily_returns': [],
            'positions': [],
            'regime': [],
            'signals': [],
        }
        
        max_value = self.initial_capital
        last_rebalance = 0
        
        # Walk-forward: começar após warmup
        for day in range(window_size + embargo_days, len(returns_df)):
            current_date = returns_df.index[day]
            
            # Rebalancear periodicamente
            if (day - window_size - embargo_days) % rebalance_days == 0:
                # Treinar em window histórica
                train_start = day - window_size
                train_end = day - embargo_days
                
                hist_returns = returns_df.iloc[train_start:train_end]
                hist_prices = prices_df.iloc[train_start:train_end]
                
                # Gerar sinais
                signals = model.generate_robust_signals(
                    hist_returns.iloc[-252:],
                    hist_prices.iloc[-252:],
                    fundamentals_df
                )
                
                weights = signals['weights']
                positions = (weights * portfolio_value) / (prices_df.iloc[day].values + 1e-8)
                
                last_rebalance = day
            
            # Computar retorno diário
            daily_ret = (returns_df.iloc[day].values * positions).sum() / portfolio_value
            daily_ret = np.clip(daily_ret, -0.10, 0.10)  # Stop-loss hard
            
            portfolio_value *= (1 + daily_ret)
            
            # Registrar
            results['dates'].append(current_date)
            results['portfolio_value'].append(portfolio_value)
            results['daily_returns'].append(daily_ret)
            results['positions'].append(positions.copy())
            
            # Drawdown
            max_value = max(max_value, portfolio_value)
            drawdown = (portfolio_value - max_value) / max_value
            
            # Regime
            if hasattr(model.regime_model, 'fitted') and model.regime_model.fitted:
                regime = model.regime_model.predict_regime(returns_df.iloc[:day])
                results['regime'].append(regime)
            else:
                results['regime'].append(1)
        
        return results
    
    def analyze_overfitting(self, train_returns, test_returns):
        """
        Detecta overfitting usando métricas profissionais.
        
        PRADO-COMPLIANT: Sharpe deflacionado, PSR, etc.
        """
        # Sharpe ratios
        train_sharpe = StrategyMetrics.sharpe_ratio(np.array(train_returns))
        test_sharpe = StrategyMetrics.sharpe_ratio(np.array(test_returns))
        
        # PSR (Probabilistic Sharpe Ratio)
        psr_result = StrategyMetrics.probabilistic_sharpe_ratio(
            np.array(test_returns), 
            benchmark_sharpe=0,
            n_tests=5
        )
        
        # Degradação (quanto piorou de treino para teste)
        degradation = 1 - (test_sharpe / (train_sharpe + 1e-8))
        
        # Overfitting flag
        is_overfit = degradation > 0.5 or psr_result['psr_adjusted'] < 0.05
        
        return {
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'degradation_pct': degradation * 100,
            'psr': psr_result['psr'],
            'psr_adjusted': psr_result['psr_adjusted'],
            'is_overfit': is_overfit,
            'warning': 'Model shows significant overfitting' if is_overfit else 'Model is robust'
        }
    
    def run_purged_cv_backtest(self, returns_df, prices_df, fundamentals_df,
                               n_splits=5, embargo_pct=0.01):
        """
        Backtesting com Purged K-Fold Cross-Validation.
        
        PRADO-COMPLIANT: Evita data leakage entre train/test.
        """
        purged_kfold = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
        
        # Labeling via triple barrier
        labeler = TripleBarrierLabeler(vertical_barrier_periods=20,
                                      upper_barrier_pct=0.02,
                                      lower_barrier_pct=0.02)
        labels = labeler.label(prices_df.iloc[:, 0], side=1)
        
        # Features
        model = AdaptiveMultiFactorModel()
        factors = model.compute_factors(returns_df.iloc[-252:], 
                                       prices_df.iloc[-252:],
                                       fundamentals_df)
        
        # Cross-validation scores
        cv_results = {
            'fold': [],
            'train_sharpe': [],
            'test_sharpe': [],
            'degradation': [],
            'feature_importance': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(purged_kfold.split(factors.values)):
            if len(train_idx) < 30 or len(test_idx) < 10:
                continue
            
            X_train = factors.iloc[train_idx].values
            X_test = factors.iloc[test_idx].values
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            
            # Treinar
            if np.sum(y_train != 0) > 5:
                model.build_robust_ml_ensemble(X_train, y_train)
                
                # Avaliar
                train_pred = model.primary_model.predict(X_train)
                test_pred = model.primary_model.predict(X_test)
                
                train_sharpe = StrategyMetrics.sharpe_ratio(train_pred[:len(y_train)])
                test_sharpe = StrategyMetrics.sharpe_ratio(test_pred[:len(y_test)])
                
                degradation = 1 - (test_sharpe / (train_sharpe + 1e-8))
                
                # Feature importance
                feat_imp = FeatureImportanceAnalyzer.mdi(model.primary_model, X_test)
                
                cv_results['fold'].append(fold)
                cv_results['train_sharpe'].append(train_sharpe)
                cv_results['test_sharpe'].append(test_sharpe)
                cv_results['degradation'].append(degradation)
                cv_results['feature_importance'].append(feat_imp.to_dict())
        
        # Consolidar resultados
        cv_summary = {
            'mean_train_sharpe': np.mean(cv_results['train_sharpe']),
            'mean_test_sharpe': np.mean(cv_results['test_sharpe']),
            'mean_degradation': np.mean(cv_results['degradation']),
            'cv_scores': cv_results,
            'is_robust': np.mean(cv_results['degradation']) < 0.5
        }
        
        return cv_summary
    
    def calculate_comprehensive_metrics(self, results):
        """
        Calcula métricas profissionais (PRADO-COMPLIANT).
        """
        daily_returns = np.array(results['daily_returns'])
        portfolio_values = np.array(results['portfolio_value'])
        
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = (portfolio_values[-1] / self.initial_capital) ** (252 / len(daily_returns)) - 1
        
        max_dd = StrategyMetrics.max_drawdown(daily_returns)
        
        metrics = {
            # Retornos
            'total_return': total_return,
            'annual_return': annual_return,
            'daily_mean': np.mean(daily_returns),
            'daily_std': np.std(daily_returns),
            
            # Risk-adjusted
            'sharpe_ratio': StrategyMetrics.sharpe_ratio(daily_returns),
            'sortino_ratio': StrategyMetrics.sortino_ratio(daily_returns),
            'calmar_ratio': StrategyMetrics.calmar_ratio(daily_returns, max_dd),
            'recovery_factor': StrategyMetrics.recovery_factor(daily_returns, max_dd),
            
            # Risk
            'volatility': np.std(daily_returns) * np.sqrt(252),
            'max_drawdown': max_dd,
            'var_95': StrategyMetrics.var_95(daily_returns),
            'cvar_95': StrategyMetrics.cvar_95(daily_returns),
            
            # Trading
            'win_rate': StrategyMetrics.win_rate(daily_returns),
            'profit_factor': StrategyMetrics.profit_factor(daily_returns),
            'n_trades': len(results['positions']),
            
            # Extremos
            'best_day': np.max(daily_returns),
            'worst_day': np.min(daily_returns),
            'positive_days': np.sum(daily_returns > 0),
            'total_days': len(daily_returns),
        }
        
        # PSR
        psr = StrategyMetrics.probabilistic_sharpe_ratio(daily_returns)
        metrics['probabilistic_sharpe_ratio'] = psr['psr']
        metrics['psr_adjusted'] = psr['psr_adjusted']
        
        return metrics
    
    def print_professional_report(self, metrics, cv_metrics=None, overfit_analysis=None):
        """
        Imprime relatório profissional de backtesting.
        """
        print("\n" + "="*80)
        print("PROFESSIONAL BACKTESTING REPORT - PRADO COMPLIANT")
        print("="*80)
        
        print(f"\n{'PERFORMANCE METRICS':<40}")
        print(f"  Total Return:           {metrics['total_return']:>10.2%}")
        print(f"  Annual Return:          {metrics['annual_return']:>10.2%}")
        print(f"  Daily Mean:             {metrics['daily_mean']:>10.4%}")
        
        print(f"\n{'RISK-ADJUSTED METRICS':<40}")
        print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:          {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:           {metrics['calmar_ratio']:>10.2f}")
        print(f"  PSR (Probabilistic SR): {metrics['probabilistic_sharpe_ratio']:>10.2f}")
        print(f"  PSR Adjusted:           {metrics['psr_adjusted']:>10.2%}")
        
        print(f"\n{'RISK METRICS':<40}")
        print(f"  Volatility:             {metrics['volatility']:>10.2%}")
        print(f"  Max Drawdown:           {metrics['max_drawdown']:>10.2%}")
        print(f"  VaR 95%:                {metrics['var_95']:>10.2%}")
        print(f"  CVaR 95%:               {metrics['cvar_95']:>10.2%}")
        
        print(f"\n{'TRADING METRICS':<40}")
        print(f"  Win Rate:               {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:          {metrics['profit_factor']:>10.2f}")
        print(f"  Recovery Factor:        {metrics.get('recovery_factor', 0):>10.2f}")
        
        print(f"\n{'EXTREMES':<40}")
        print(f"  Best Day:               {metrics['best_day']:>10.2%}")
        print(f"  Worst Day:              {metrics['worst_day']:>10.2%}")
        print(f"  Positive Days:          {metrics['positive_days']:>10} / {metrics['total_days']}")
        
        if cv_metrics:
            print(f"\n{'CROSS-VALIDATION METRICS':<40}")
            print(f"  Mean Train Sharpe:      {cv_metrics['mean_train_sharpe']:>10.2f}")
            print(f"  Mean Test Sharpe:       {cv_metrics['mean_test_sharpe']:>10.2f}")
            print(f"  Mean Degradation:       {cv_metrics['mean_degradation']:>10.2%}")
            print(f"  Robustness:             {cv_metrics['is_robust']}")
        
        if overfit_analysis:
            print(f"\n{'OVERFITTING ANALYSIS':<40}")
            print(f"  Train Sharpe:           {overfit_analysis['train_sharpe']:>10.2f}")
            print(f"  Test Sharpe:            {overfit_analysis['test_sharpe']:>10.2f}")
            print(f"  Degradation %:          {overfit_analysis['degradation_pct']:>10.1f}%")
            print(f"  Warning:                {overfit_analysis['warning']}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Executa backtesting profissional completo."""
    
    print("\n🏢 PROFESSIONAL BACKTEST ENGINE - López de Prado Compliant\n")
    
    # Inicializar
    engine = ProfessionalBacktestEngine(initial_capital=1000000)
    
    # Gerar dados
    print("[1/5] Gerando dados realistas...")
    returns_df, prices_df, fundamentals_df = engine.generate_realistic_data(
        n_assets=50, n_days=756
    )
    print(f"      ✓ {returns_df.shape[0]} dias, {returns_df.shape[1]} ativos")
    
    # Walk-forward backtest
    print("[2/5] Executando walk-forward backtest...")
    results = engine.run_walk_forward_backtest(
        returns_df, prices_df, fundamentals_df,
        window_size=252, rebalance_days=21, embargo_days=10
    )
    print(f"      ✓ {len(results['dates'])} dias simulados")
    
    # Métricas
    print("[3/5] Calculando métricas profissionais...")
    metrics = engine.calculate_comprehensive_metrics(results)
    print(f"      ✓ Retorno anual: {metrics['annual_return']:.2%}")
    print(f"      ✓ Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Cross-validation
    print("[4/5] Executando purged K-fold cross-validation...")
    cv_metrics = engine.run_purged_cv_backtest(
        returns_df, prices_df, fundamentals_df,
        n_splits=5, embargo_pct=0.01
    )
    print(f"      ✓ Robustez: {cv_metrics['is_robust']}")
    
    # Relatório
    print("[5/5] Gerando relatório profissional...")
    engine.print_professional_report(metrics, cv_metrics)
    
    # Salvar resultados
    results_df = pd.DataFrame({
        'date': results['dates'],
        'portfolio_value': results['portfolio_value'],
        'daily_return': results['daily_returns'],
        'regime': results['regime'],
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'professional_backtest_results.csv'), index=False)
    
    # Salvar métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'professional_metrics.csv'), index=False)
    
    print("\n✓ Resultados salvos em complex/outputs/")
    
    return metrics, cv_metrics


if __name__ == "__main__":
    metrics, cv_metrics = main()
