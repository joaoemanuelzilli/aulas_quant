# aulas_quant
Repositório das Aulas do Núcleo Quant

Aula 1 - 14/03/2026
- Apresentação da estrutura do núcleo.;
- Ambiente de trabalho (VS Code, Git);
- Introdução a pandas e NumPy;
- Importação e manipulação de séries de preços;
- Cálculo de retornos simples e log-retornos.

Aula 2 - 28/03/2026
- yfinance & APIs;
- Dados faltantes;
- Visualização com Matplotlib/Plotly;
- Análise exploratória de ativos.

Aula 3 - 04/04/2026

- Estatística descritiva;
- Distribuições (normal, t-Student, caudas pesadas);
- Testes de hipótese;
- Correlação.

Aula 4 - 11/04/2026

- Estacionariedade & testes ADF;
- ACF/PACF;
- Modelos ARIMA · GARCH;
- Aplicação em dados reais.

Aula 5 - 18/04/2026

- VaR: paramétrico, histórico e Monte Carlo;
- CVaR / Expected Shortfall;
- Drawdown máximo;
- Métricas de cauda.

Aula 6 - 02/05/2026

- Teoria de Markowitz;
- Fronteira Eficiente;
- Diversificação;
- Stress Testing e Análise de Cenários. 

Aula 7 - 28/05/2026

- Fundamentos de backtesting;
- Métricas de performance: Sharpe, Sortino, Calmar, Treynor;
- Vieses comuns: look-ahead bias, survivorship bias, overfitting;
- Introdução ao framework backtrader;
- Otimização de parâmetros no backtrader com separação in/out-of-sample.

Scripts da aula (ver [aula7/README.md](aula7/README.md)):

- `01_lookahead_bias.py` — look-ahead bias;
- `02_survivorship_bias.py` — survivorship bias;
- `03_overfitting.py` — overfitting;
- `04_backtrader_intro.py` — estrutura básica do backtrader e primeiro backtest;
- `05_backtrader_otimizacao.py` — otimização com treino/teste.

Projeto - complex/

Estratégia Adaptive Multi-Factor Regime-Switching (ver [complex/README.md](complex/README.md)):

- Hidden Markov Model para detecção de regime;
- Múltiplos fatores (value, momentum, quality, low-vol, entre outros);
- Ensemble de ML (Gradient Boosting + Ridge);
- Engine de backtest, relatórios em LaTeX/PDF e documentação em `complex/docs/`.

Os temas detalhados de cada aula estão em [README_temas.md](README_temas.md).
