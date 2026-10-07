"""Números citados no texto que não aparecem nas saídas impressas."""
import numpy as np
import pandas as pd

from l03_custos import resultado, sharpe
from l04_bot import b, cor, curva

# dado sujo: só existe na cópia bruta (Capítulo 2)
o = pd.read_csv("../../dados/bruto/petr4_ohlcv.csv", index_col=0,
                parse_dates=True)
limpo = pd.read_csv("../../dados/petr4_ohlcv.csv", index_col=0,
                    parse_dates=True)
parado = (o.Volume == 0) & (o.High == o.Low)
print("barras no bruto:", len(o), "| no limpo:", len(limpo),
      "| removidas:", len(o) - len(limpo),
      "| volume zero no limpo:", (limpo.Volume == 0).sum())
print("bruto: barras com volume zero e OHLC iguais:", parado.sum())

# liquidez: volume financeiro mediano de 2025, R$ milhões
vol = pd.read_csv("../../dados/bruto/volume.csv", index_col=0,
                  parse_dates=True).loc["2025"]
tela = pd.read_csv("../../dados/precos_sem_ajuste.csv", index_col=0,
                   parse_dates=True).loc["2025"]
fin = (vol * tela[vol.columns]).median() / 1e6
print(fin.sort_values(ascending=False).round(0).to_dict())

compras = [e for e in cor.extrato if e[2] == "compra"]
abaixo = sum(b.Low[e[0]] < round(e[5] * 0.95, 2) for e in compras)
acima = sum(b.High[e[0]] > round(e[5] * 1.08, 2) for e in compras)
na_abertura = sum(abs(e[5] / b.Open[e[0]] - 1.0005) < 1e-9
                  for e in compras)
print(f"entradas: {len(compras)}; mínima do dia abaixo do stop: "
      f"{abaixo}; máxima acima do alvo: {acima}; "
      f"executadas na abertura com slippage: {na_abertura}")
print("ordens rejeitadas:",
      sum(x.estado == "rejeitada" for x in cor.ordens))

# o stop de 22/02/2021
ant, dia = pd.Timestamp("2021-02-19"), pd.Timestamp("2021-02-22")
compra = [e for e in cor.extrato if e[0] == ant][0]
venda = [e for e in cor.extrato if e[0] == dia][0]
gat = [x for x in cor.ordens if x.id == venda[1]][0].preco
lim = [x for x in cor.ordens if x.id == compra[1]][0].preco
print(f"compra {ant:%d/%m}: limite {lim}, abertura "
      f"{b.Open[ant]:.4f}, executada a {compra[5]:.4f}")
print(f"abertura {dia:%d/%m} contra fechamento anterior: "
      f"{b.Open[dia] / b.Close[ant] - 1:.1%}")
print(f"stop: gatilho {gat}, executado a {venda[5]:.4f} "
      f"({venda[5] / gat - 1:.1%} do gatilho)")
print(f"perda da operação: {venda[5] / compra[5] - 1:.1%}; "
      f"patrimônio no dia: {curva[dia] / curva[ant] - 1:.2%}")
print(f"posição / patrimônio na compra: "
      f"{compra[4] * compra[5] / curva[ant]:.1%}")

# exposição média e operações do bot
pos = pd.Series({e[0]: e[7] for e in cor.extrato})
pos = pos.reindex(curva.index).ffill().fillna(0)
print(f"dias com posição: {(pos > 0).mean():.0%} de {len(pos)}")
print(f"retorno do bot: {curva.iloc[-1] / 1e5 - 1:.1%}")

# estratégia de momentum: anos de dados e erro-padrão do Sharpe
for nome, (bruto, giro) in resultado.items():
    anos = len(bruto) / 252
    s = sharpe(bruto)
    ep = np.sqrt((1 + s ** 2 / 504) / anos)   # Lo, SR anual
    print(f"{nome}: {anos:.1f} anos, Sharpe bruto {s:.2f}, "
          f"erro-padrão {ep:.2f}, bruto {bruto.mean() * 252:.2%},"
          f" giro por dia {giro.mean():.2%},"
          f" giro por rebalanceamento {giro[giro > 0].mean():.1%}")

# Kelly contínuo do exemplo do texto
mu, sig = 0.06, 0.20
print(f"Kelly contínuo: f* = {mu / sig ** 2:.2f}; "
      f"g* = {mu ** 2 / (2 * sig ** 2):.3f}; "
      f"meio Kelly: {0.75 * mu ** 2 / (2 * sig ** 2):.4f}")
