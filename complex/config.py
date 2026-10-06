"""
CONFIGURAÇÕES DO MODELO - Adaptive Multi-Factor Regime-Switching

Este arquivo contém todos os parâmetros configuráveis da estratégia.
Modifique aqui para ajustar comportamento sem editar código.
"""

# ============================================================================
# PARÂMETROS DO MODELO
# ============================================================================

# Janela histórica para cálculos (em dias)
LOOKBACK_WINDOW = 252  # 1 ano

# Horizonte de previsão (em dias)
FORECAST_HORIZON = 20

# ============================================================================
# PARÂMETROS DO HMM (Hidden Markov Model)
# ============================================================================

# Número de regimes/estados
HMM_N_STATES = 3

# Número máximo de iterações para treinar HMM
HMM_MAX_ITER = 1000

# Tipo de covariância do HMM
HMM_COVARIANCE_TYPE = "full"  # "full", "diag", "spherical", "tied"

# ============================================================================
# PARÂMETROS DO GRADIENT BOOSTING
# ============================================================================

GB_N_ESTIMATORS = 200      # Número de árvores
GB_LEARNING_RATE = 0.05    # Taxa de aprendizado
GB_MAX_DEPTH = 5           # Profundidade máxima das árvores
GB_SUBSAMPLE = 0.8         # Fração de amostras por iteração
GB_MIN_SAMPLES_LEAF = 5    # Mínimo de amostras por folha

# ============================================================================
# PARÂMETROS DO RIDGE REGRESSION
# ============================================================================

RIDGE_ALPHA = 1.0          # Força da regularização L2

# ============================================================================
# ENSEMBLE ML - PESOS POR REGIME
# ============================================================================

# Pesos do ensemble em cada regime
# [Gradient Boosting weight, Ridge weight]
ML_WEIGHTS = {
    'bull': [0.6, 0.4],      # Bull market: confia mais em GB (não-linear)
    'bear': [0.4, 0.6],      # Bear market: confia mais em Ridge (estável)
    'highvol': [0.5, 0.5],   # High volatility: equilibrado
}

# ============================================================================
# PARÂMETROS DE LEVERAGE - DINÂMICOS POR REGIME
# ============================================================================

LEVERAGE = {
    'bull': 1.3,      # 130% - Agressivo, tomar risco em bull markets
    'bear': 0.7,      # 70% - Defensivo, reduzir exposição em bears
    'highvol': 1.0,   # 100% - Equilibrado em alta volatilidade
}

# ============================================================================
# PARÂMETROS DE CONCENTRAÇÃO - PESO MÁXIMO POR ATIVO
# ============================================================================

MAX_WEIGHT = {
    'bull': 0.08,     # Máximo 8% por ativo em bull
    'bear': 0.05,     # Máximo 5% por ativo em bear (mais diversificado)
    'highvol': 0.06,  # Máximo 6% por ativo em high vol
}

# ============================================================================
# PARÂMETROS DE OTIMIZAÇÃO DE PORTFOLIO
# ============================================================================

# Volatilidade alvo anualizada
TARGET_VOLATILITY = 0.08  # 8% a.a.

# Limite máximo de turnover (rotação de portfolio)
MAX_TURNOVER = 0.30  # 30% do portfolio pode ser rotacionado por rebalance

# Limite de loss diário (stop-loss global)
MAX_DAILY_LOSS = -0.10  # -10% por dia

# ============================================================================
# PARÂMETROS DE BACKTEST
# ============================================================================

# Capital inicial do backtest
INITIAL_CAPITAL = 1000000  # $1,000,000

# Frequência de rebalanceamento (em dias de trading)
REBALANCE_DAYS = 21  # Mensalmente (21 dias de trading ≈ 1 mês)

# Período de warm-up (não tradamos nos primeiros N dias)
WARMUP_PERIOD = 252  # 1 ano de dados históricos antes de começar

# ============================================================================
# PARÂMETROS DE CUSTOS DE TRANSAÇÃO
# ============================================================================

# Custo de transação por trade (em basis points)
TRANSACTION_COST_BPS = 10  # 10 bps = 0.001%

# Slippage (diferença entre preço teórico e executado)
SLIPPAGE_BPS = 5  # 5 bps

# Custo fixo por rebalanceamento
FIXED_REBALANCE_COST = 0  # $0 (pode adicionar se necessário)

# ============================================================================
# PARÂMETROS DE GESTÃO DE RISCO
# ============================================================================

# Limite de drawdown que ativa redução de leverage
DRAWDOWN_TRIGGER = -0.20  # -20% de drawdown

# Novo leverage quando drawdown é acionado
LEVERAGE_WHEN_DRAWDOWN = 0.5  # Reduzir para 50% da exposição

# ============================================================================
# VALIDAÇÃO CRUZADA
# ============================================================================

# Tamanho da janela de teste em walk-forward (em dias)
WALK_FORWARD_TEST_SIZE = 20  # 20 dias de teste

# Frequência de retraining
RETRAIN_FREQUENCY = 1  # Retreinar a cada 1 dia

# ============================================================================
# FATORES UTILIZADOS
# ============================================================================

# Lista de fatores e seus pesos na média
FACTORS = {
    'value': 1.0,         # Book-to-Market
    'momentum': 1.0,      # Retorno 12m
    'quality': 1.0,       # ROE
    'low_vol': 1.0,       # Inverso volatilidade
    'profitability': 1.0, # EBIT / Assets
    'growth': 1.0,        # Crescimento receita
    'mean_reversion': 1.0 # Reverter à média
}

# ============================================================================
# PARÂMETROS PRADO-COMPLIANT (Advanced Financial Machine Learning)
# ============================================================================

# Embargo period: % de dados removidos entre train e test
EMBARGO_PCT = 0.01  # 1% de embargo (protege contra look-ahead bias)

# Triple Barrier Method (robust labeling)
TRIPLE_BARRIER = {
    'upper_barrier_pct': 0.02,        # Upper barrier: +2%
    'lower_barrier_pct': 0.02,        # Lower barrier: -2%
    'vertical_barrier_periods': 20,   # Vertical barrier: 20 dias
}

# Meta-labeling configuration
META_LABELING = {
    'enabled': True,
    'return_threshold': 0.005,  # Threshold para label meta
    'model_type': 'logistic',   # Tipo de modelo meta
    'cv_folds': 3,              # Cross-validation folds
}

# Purged K-Fold Cross-Validation
PURGED_KFOLD = {
    'n_splits': 5,              # Número de folds
    'embargo_pct': EMBARGO_PCT, # Embargo period
}

# Detecção de Overfitting
OVERFITTING_DETECTION = {
    'enabled': True,
    'train_test_ratio_threshold': 2.0,  # Sharpe train/test ratio máximo
    'psr_threshold': 0.05,               # PSR adjusted p-value
    'sharpe_degradation_threshold': 0.5, # Máximo degradação permitida
}

# Feature Importance Analysis
FEATURE_IMPORTANCE = {
    'method': 'mdi_mda',  # Mean Decrease in Impurity + Mean Decrease in Accuracy
    'n_permutations': 100, # Permutações para MDA
    'threshold': 0.05,     # Mínimo de importância para reter feature
}

# Sharpe Ratio Deflation (correção para múltiplos testes)
SHARPE_DEFLATION = {
    'enabled': True,
    'n_tests': 10,  # Número estimado de testes realizados
    'method': 'bonferroni',  # 'bonferroni' ou 'holm'
}

# Structural Break Detection
STRUCTURAL_BREAK = {
    'enabled': True,
    'window_size': 60,      # Tamanho da janela para detectar breaks
    'threshold_std': 2.0,   # Múltiplos de desvio padrão
}

# ============================================================================
# LOGGING E DEBUG
# ============================================================================

# Nível de verbosidade
VERBOSE = True

# Salvar detalhes de cada rebalanceamento
SAVE_REBALANCE_DETAILS = True

# ============================================================================
# EXEMPLO DE USO
# ============================================================================

"""
import config

# Usar configurações
engine = BacktestEngine(
    initial_capital=config.INITIAL_CAPITAL,
    rebalance_freq=config.REBALANCE_DAYS
)

model = AdaptiveMultiFactorModel(
    lookback_window=config.LOOKBACK_WINDOW,
    forecast_horizon=config.FORECAST_HORIZON
)

# Para mudar parâmetros dinamicamente:
config.TARGET_VOLATILITY = 0.10  # Aumentar target vol para 10%
config.LEVERAGE['bull'] = 1.5    # Mais agressivo em bull markets
"""

# ============================================================================
# PERFIS PRÉ-CONFIGURADOS
# ============================================================================

PROFILES = {
    'conservative': {
        'leverage': {'bull': 1.0, 'bear': 0.5, 'highvol': 0.7},
        'max_weight': {'bull': 0.06, 'bear': 0.03, 'highvol': 0.05},
        'target_volatility': 0.06,
    },
    'moderate': {
        'leverage': {'bull': 1.2, 'bear': 0.7, 'highvol': 1.0},
        'max_weight': {'bull': 0.07, 'bear': 0.04, 'highvol': 0.06},
        'target_volatility': 0.08,
    },
    'aggressive': {
        'leverage': {'bull': 1.5, 'bear': 0.9, 'highvol': 1.2},
        'max_weight': {'bull': 0.10, 'bear': 0.06, 'highvol': 0.08},
        'target_volatility': 0.12,
    },
}

# Para usar um perfil pré-configurado:
# profile = PROFILES['moderate']
# LEVERAGE = profile['leverage']
# MAX_WEIGHT = profile['max_weight']
# TARGET_VOLATILITY = profile['target_volatility']
