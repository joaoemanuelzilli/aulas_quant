"""
AULA 7 - Backtrader: Otimização de Parâmetros
================================================
Mostra como usar cerebro.optstrategy() para testar múltiplas
combinações de parâmetros automaticamente — e como evitar
o overfitting nesse processo.

Divisão in-sample / out-of-sample dentro do próprio Backtrader.
"""

import os
import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("aula7/outputs", exist_ok=True)

# --------------------------------------------------------------------------
# 1. ESTRATÉGIA PARAMETRIZADA
# --------------------------------------------------------------------------
class CruzamentoMedias(bt.Strategy):
    params = (
        ("periodo_rapido", 20),
        ("periodo_lento",  50),
    )

    def __init__(self):
        self.sma_rapida = bt.ind.SMA(period=self.p.periodo_rapido)
        self.sma_lenta  = bt.ind.SMA(period=self.p.periodo_lento)
        self.crossover  = bt.ind.CrossOver(self.sma_rapida, self.sma_lenta)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.sell()


# --------------------------------------------------------------------------
# 2. FUNÇÃO AUXILIAR: montar e rodar o cerebro
# --------------------------------------------------------------------------
def rodar_backtest(df, periodo_rapido, periodo_lento, capital=100_000, verbose=False):
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(CruzamentoMedias,
                        periodo_rapido=periodo_rapido,
                        periodo_lento=periodo_lento)
    cerebro.broker.setcash(capital)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.105, annualize=True)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat   = results[0]
    sharpe  = strat.analyzers.sharpe.get_analysis().get("sharperatio") or 0
    retorno = (cerebro.broker.getvalue() / capital - 1) * 100

    if verbose:
        print(f"  SMA({periodo_rapido:3d}, {periodo_lento:3d}) → Retorno: {retorno:6.2f}%  Sharpe: {sharpe:.3f}")

    return retorno, sharpe


# --------------------------------------------------------------------------
# 3. DADOS E DIVISÃO IN-SAMPLE / OUT-OF-SAMPLE
# --------------------------------------------------------------------------
print("Baixando dados...")
df_raw = yf.download("ITUB4.SA", start="2015-01-01", end="2024-01-01", auto_adjust=True)
df_raw.index = pd.to_datetime(df_raw.index)
df_raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_raw.columns]
if "adj close" in df_raw.columns:
    df_raw = df_raw.drop(columns=["adj close"])

split = int(len(df_raw) * 0.70)
df_treino = df_raw.iloc[:split].copy()
df_teste  = df_raw.iloc[split:].copy()

print(f"Treino : {df_treino.index[0].date()} → {df_treino.index[-1].date()}")
print(f"Teste  : {df_teste.index[0].date()}  → {df_teste.index[-1].date()}")
print()

# --------------------------------------------------------------------------
# 4. GRID SEARCH NO TREINO
# --------------------------------------------------------------------------
periodos_rapidos = [10, 20, 30, 40]
periodos_lentos  = [50, 80, 100, 150, 200]

print("Otimizando no TREINO (in-sample)...")
resultados = []

for pr in periodos_rapidos:
    for pl in periodos_lentos:
        if pr >= pl:
            continue
        ret, sharpe = rodar_backtest(df_treino, pr, pl)
        resultados.append({
            "rapido": pr, "lento": pl,
            "ret_treino": ret, "sharpe_treino": sharpe
        })

df_res = pd.DataFrame(resultados).sort_values("sharpe_treino", ascending=False)

melhor = df_res.iloc[0]
print(f"\nMelhor no treino: SMA({int(melhor['rapido'])}, {int(melhor['lento'])})")
print(f"  Sharpe treino: {melhor['sharpe_treino']:.3f}  |  Retorno: {melhor['ret_treino']:.2f}%")

# --------------------------------------------------------------------------
# 5. VALIDAR NO TESTE (out-of-sample)
# --------------------------------------------------------------------------
print("\nAplicando os top-3 parâmetros no TESTE (out-of-sample)...")
print("-" * 60)

for _, row in df_res.head(3).iterrows():
    pr, pl = int(row["rapido"]), int(row["lento"])
    ret_teste, sharpe_teste = rodar_backtest(df_teste, pr, pl, verbose=True)

# Buy & Hold no teste
bh_teste = (df_teste["close"].iloc[-1] / df_teste["close"].iloc[0] - 1) * 100
print(f"\nBuy & Hold no teste: {bh_teste:.2f}%")

# --------------------------------------------------------------------------
# 6. VISUALIZAÇÃO: heatmap de Sharpe no treino
# --------------------------------------------------------------------------
pivot = df_res.pivot(index="rapido", columns="lento", values="sharpe_treino")

plt.figure(figsize=(10, 6))
plt.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
plt.colorbar(label="Sharpe Ratio (treino)")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)),   pivot.index)
plt.xlabel("Período lento")
plt.ylabel("Período rápido")
plt.title("Heatmap de Sharpe — Otimização in-sample\n(cuidado: escolher o melhor aqui é a origem do overfitting!)")
plt.tight_layout()
plt.savefig("aula7/outputs/05_heatmap_sharpe.png", dpi=150)
plt.close()

print("""
LIÇÃO IMPORTANTE:
-----------------
O heatmap mostra os Sharpes no TREINO. Antes de confiar
na melhor célula, sempre verifique se ela se mantém boa
no TESTE. Se o desempenho despencar fora da amostra,
o modelo está overfit.
""")
