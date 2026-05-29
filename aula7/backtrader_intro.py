"""
AULA 7 - Introdução ao Backtrader
====================================
Framework profissional para backtesting em Python.

Conceitos principais:
  - Cerebro      : o "motor" que coordena tudo
  - Strategy     : onde você escreve a lógica de trading
  - Data Feed    : os dados históricos de preço
  - Broker       : simula a corretora (capital, ordens, comissão)
  - Analyzers    : métricas automáticas (Sharpe, Drawdown, etc.)

Instalação:
  pip install backtrader
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")


def baixar_cdi_diario(data_inicial, data_final):
    """
    Baixa a série diária do CDI no Banco Central.
    SGS 12 já vem como taxa percentual ao dia.
    """
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
    params = {
        "formato": "json",
        "dataInicial": data_inicial.strftime("%d/%m/%Y"),
        "dataFinal": data_final.strftime("%d/%m/%Y"),
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()

    cdi = pd.DataFrame(resp.json())
    cdi["data"] = pd.to_datetime(cdi["data"], dayfirst=True)
    cdi["valor"] = pd.to_numeric(cdi["valor"]) / 100

    return cdi.set_index("data")["valor"].sort_index()


def calcular_sharpe_com_cdi(retornos_diarios):
    """
    Calcula o Sharpe fora do Backtrader usando o CDI histórico diário
    como retorno livre de risco variável no tempo.
    """
    retornos = pd.Series(retornos_diarios)
    retornos.index = pd.to_datetime(retornos.index)
    retornos = retornos.sort_index()

    cdi = baixar_cdi_diario(retornos.index.min(), retornos.index.max())
    cdi_alinhado = cdi.reindex(retornos.index).fillna(0)

    excesso = retornos - cdi_alinhado
    desvio = excesso.std()

    if desvio == 0 or pd.isna(desvio):
        return None

    return excesso.mean() / desvio * np.sqrt(252)

# --------------------------------------------------------------------------
# 1. ESTRATÉGIA: cruzamento de médias móveis
# --------------------------------------------------------------------------
class CruzamentoMedias(bt.Strategy):
    """
    Compra quando SMA rápida cruza acima da SMA lenta.
    Vende quando SMA rápida cruza abaixo da SMA lenta.
    """

    # Parâmetros configuráveis — fácil de testar variações
    params = (
        ("periodo_rapido", 20),
        ("periodo_lento",  50),
    )

    def __init__(self):
        # Indicadores calculados automaticamente pelo Backtrader
        self.sma_rapida = bt.ind.SMA(period=self.p.periodo_rapido)
        self.sma_lenta  = bt.ind.SMA(period=self.p.periodo_lento)

        # CrossOver retorna +1 quando rápida cruza acima, -1 quando cruza abaixo
        self.crossover = bt.ind.CrossOver(self.sma_rapida, self.sma_lenta)

    def next(self):
        # self.position retorna a posição atual (0 = sem posição)
        if not self.position:
            if self.crossover > 0:          # cruzamento para cima → compra
                self.buy()
        else:
            if self.crossover < 0:          # cruzamento para baixo → vende
                self.sell()

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f"  Trade fechado | P&L bruto: R${trade.pnl:8.2f} | P&L líquido: R${trade.pnlcomm:8.2f}")

    def notify_order(self, order):
        if order.status == order.Completed:
            tipo = "COMPRA" if order.isbuy() else "VENDA"
            dt = bt.num2date(order.executed.dt).date()
            print(f"  {dt} | {tipo} | Preço: {order.executed.price:.2f}")


# --------------------------------------------------------------------------
# 2. PREPARAR DADOS
# --------------------------------------------------------------------------
print("Baixando dados...")
df = yf.download("VALE3.SA", start="2020-01-01", end="2026-01-01", auto_adjust=True)
df.index = pd.to_datetime(df.index)

# Backtrader espera colunas com nomes específicos (case-insensitive)
df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
if "adj close" in df.columns:
    df = df.drop(columns=["adj close"])

print(f"Período: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} pregões)\n")

# --------------------------------------------------------------------------
# 3. CONFIGURAR O CEREBRO
# --------------------------------------------------------------------------
cerebro = bt.Cerebro()

# 3a. Adicionar dados
feed = bt.feeds.PandasData(dataname=df)
cerebro.adddata(feed)

# 3b. Adicionar a estratégia
cerebro.addstrategy(CruzamentoMedias, periodo_rapido=20, periodo_lento=50)

# 3c. Capital inicial e comissão (0.1% por ordem, similar à B3)
cerebro.broker.setcash(100_000)
cerebro.broker.setcommission(commission=0.001)

# 3d. Sizer: aloca 95% do capital disponível em cada compra
cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

# 3e. Analisadores de métricas
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.145, annualize=True)
cerebro.addanalyzer(bt.analyzers.DrawDown,    _name="dd")
cerebro.addanalyzer(bt.analyzers.Returns,     _name="returns")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="retornos_diarios", timeframe=bt.TimeFrame.Days)

# --------------------------------------------------------------------------
# 4. EXECUTAR
# --------------------------------------------------------------------------
capital_inicial = cerebro.broker.getvalue()
print(f"Capital inicial: R${capital_inicial:,.2f}")
print("-" * 50)
print("Trades executados:")

results = cerebro.run()
strat   = results[0]

capital_final = cerebro.broker.getvalue()

# --------------------------------------------------------------------------
# 5. EXIBIR RESULTADOS
# --------------------------------------------------------------------------
print("-" * 50)
print("\n======== RESUMO DO BACKTEST ========")
print(f"Capital inicial : R${capital_inicial:>12,.2f}")
print(f"Capital final   : R${capital_final:>12,.2f}")
print(f"Retorno total   : {(capital_final / capital_inicial - 1) * 100:>10.2f}%")

sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
retornos_diarios = strat.analyzers.retornos_diarios.get_analysis()
dd     = strat.analyzers.dd.get_analysis()
trades = strat.analyzers.trades.get_analysis()

if sharpe is not None:
    print(f"Sharpe Ratio    : {sharpe:>10.3f}")

try:
    sharpe_cdi = calcular_sharpe_com_cdi(retornos_diarios)
    if sharpe_cdi is not None:
        print(f"Sharpe c/ CDI   : {sharpe_cdi:>10.3f}")
except Exception as e:
    print(f"Sharpe c/ CDI   : não calculado ({e})")

print(f"Drawdown máximo : {dd.get('max', {}).get('drawdown', 0):>9.2f}%")

total_trades = trades.get("total", {}).get("closed", 0)
won          = trades.get("won",   {}).get("total", 0)
lost         = trades.get("lost",  {}).get("total", 0)

if total_trades > 0:
    print(f"Total de trades : {total_trades:>10}")
    print(f"  Vencedores    : {won:>10}  ({won/total_trades*100:.1f}%)")
    print(f"  Perdedores    : {lost:>10}  ({lost/total_trades*100:.1f}%)")

print("=" * 36)

# --------------------------------------------------------------------------
# 6. PLOT
# --------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    figs = cerebro.plot(style="candle", volume=False, iplot=False)
    import os
    os.makedirs("/home/joao-ivare/Downloads/Quant/aula7/outputs", exist_ok=True)
    plt.savefig("/home/joao-ivare/Downloads/Quant/aula7/outputs/backtest_plot.png", bbox_inches="tight", dpi=150)
    print("\nPlot salvo em aula7/outputs/backtest_plot.png")
except Exception as e:
    print(f"\n(Plot ignorado: {e})")
