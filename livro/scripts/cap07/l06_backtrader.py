import backtrader as bt
import pandas as pd

import metricas as m


class CruzamentoMedias(bt.Strategy):
    params = (("rapida", 20), ("lenta", 100))

    def __init__(self):
        fecho = self.data.close
        media_r = bt.ind.SMA(fecho, period=self.p.rapida)
        media_l = bt.ind.SMA(fecho, period=self.p.lenta)
        self.acima = media_r > media_l       # linha de 0 e 1

    def next(self):                          # uma vez por barra
        if self.acima[0] and not self.position:
            self.order_target_percent(target=0.95)
        elif not self.acima[0] and self.position:
            self.close()

    def notify_order(self, ordem):           # a cada mudança
        if ordem.status in (ordem.Margin, ordem.Rejected):
            print(f"{self.data.datetime.date(0)}: ordem "
                  f"recusada ({ordem.getstatusname()})")


dados = pd.read_csv("../../dados/petr4_ohlcv.csv",
                    index_col=0, parse_dates=True)
cerebro = bt.Cerebro()
cerebro.adddata(bt.feeds.PandasData(dataname=dados))
cerebro.addstrategy(CruzamentoMedias)
cerebro.broker.setcash(100_000)
cerebro.broker.setcommission(commission=0.001)   # 0,1% por ordem
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="diario")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ops")
estrategia = cerebro.run()[0]

an = estrategia.analyzers
r = pd.Series(an.diario.get_analysis()).loc["2011":"2025"]
print(f"patrimônio final: R$ {cerebro.broker.getvalue():,.2f}")
print(f"retorno a.a. {m.retorno_anual(r):.1%}   "
      f"vol a.a. {m.vol_anual(r):.1%}   "
      f"drawdown máx. {m.drawdown_maximo(r):.1%}")

ops = an.ops.get_analysis()
fechadas = ops.total.get("closed", 0)
if fechadas:
    ganhos, perdas = ops.won.pnl, ops.lost.pnl
    print(f"operações encerradas: {fechadas}   "
          f"taxa de acerto: {ops.won.total / fechadas:.1%}")
    print(f"ganho médio: R$ {ganhos.average:,.0f}   "
          f"perda média: R$ {perdas.average:,.0f}   "
          f"payoff: {ganhos.average / -perdas.average:.2f}")
    print(f"fator de lucro: {ganhos.total / -perdas.total:.2f}")
else:
    print("nenhuma operação fechada")
