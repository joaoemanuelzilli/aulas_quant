"""Números citados no texto que não aparecem nas saídas impressas."""
import numpy as np
import pandas as pd

from l03_custos import resultado, sharpe
from l04_bot import b, cor, curva

# liquidez: volume financeiro oficial mediano de 2025, R$ milhões
fin = pd.read_csv("../../dados/volume_financeiro.csv", index_col=0,
                  parse_dates=True).loc["2025"].median() / 1e6
print(fin.sort_values(ascending=False).round(0).to_dict())
o = pd.read_csv("../../dados/petr4_ohlcv.csv", index_col=0,
                parse_dates=True)
print("barras:", len(o), "| volume zero:", (o.Volume == 0).sum())

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

# ---- contas adicionais do texto ----
from l04_bot import media, RISCO, STOP, ALVO, RECUO, PRAZO
from papel import CorretoraPapel

# uma ordem, do envio ao extrato
d0, d1 = pd.Timestamp("2019-01-21"), pd.Timestamp("2019-01-22")
q14 = cor.ordens[13].qtd
print("risco por ação até o stop:", round(STOP * round(
    b.Close[d0] * (1 - RECUO), 2), 4))
lim = round(b.Close[d0] * (1 - RECUO), 2)
print(f"21/01/2019: fechamento {b.Close[d0]:.4f}, média de 100 "
      f"dias {media[d0]:.4f}, limite {lim}, quantidade bruta "
      f"{RISCO * 1e5 / (STOP * lim):.0f}")
print(f"22/01/2019: abertura {b.Open[d1]:.4f}, mínima "
      f"{b.Low[d1]:.4f}, fechamento {b.Close[d1]:.4f}; valor "
      f"{q14 * lim:.2f}, comissão {0.0003 * q14 * lim:.2f}, caixa "
      f"{1e5 - q14 * lim * 1.0003:.2f}; stop "
      f"{round(lim * (1 - STOP), 2)}, alvo {round(lim * (1 + ALVO), 2)}")
o1 = cor.ordens[13]
print("ordem 14:", o1.lado, o1.tipo, o1.qtd, o1.preco, o1.estado,
      "| ordens 1 a 13:", {x.estado for x in cor.ordens[:13]},
      "| primeira ordem enviada em", cor.ordens[0].preco)


def rodar(c):
    """O mesmo laço de l04_bot.py, sobre a corretora c."""
    dias, curva = 0, {}
    for dia, barra in b.loc["2019":].iterrows():
        c.processar(dia, barra)
        abertas, f = c.abertas(), barra.Close
        if c.posicao == 0:
            for o in abertas:
                c.cancelar(o)
            if f > media[dia]:
                limite = round(f * (1 - RECUO), 2)
                q = int(RISCO * c.patrimonio(f)
                        / (STOP * limite) / 100) * 100
                c.enviar("compra", q, "limitada", limite)
            dias = 0
        elif not abertas:
            pe = c.extrato[-1].preco
            c.enviar("venda", c.posicao, "stop",
                     round(pe * (1 - STOP), 2))
            c.enviar("venda", c.posicao, "limitada",
                     round(pe * (1 + ALVO), 2))
        else:
            dias += 1
            if dias == PRAZO:
                for o in abertas:
                    c.cancelar(o)
                c.enviar("venda", c.posicao)
        curva[dia] = c.patrimonio(f)
    compras = sum(e.lado == "compra" for e in c.extrato)
    return curva[dia], compras


class UmCentavo(CorretoraPapel):
    """Limitada só executa se a barra passar 1 centavo do limite."""
    def _preco(self, o, bar):
        p = super()._preco(o, bar)
        if o.tipo == "limitada" and p == o.preco:
            s = 1 if o.lado == "compra" else -1
            ext = bar.Low if s == 1 else bar.High
            if s * (ext - o.preco) > -0.01 + 1e-12:
                return None
        return p


print("exercício do slippage (patrimônio final, compras):")
for sl in (0, 0.0005, 0.002, 0.005):
    print(f"  {sl * 1e4:4.0f} pb: atravessar",
          rodar(CorretoraPapel(1e5, slippage=sl)),
          "| um centavo", rodar(UmCentavo(1e5, slippage=sl)))
tempo = [(v[0] - c[0]).days for c, v in zip(cor.extrato[::2],
                                           cor.extrato[1::2])
         if v.tipo == "mercado"]
pregoes = [b.index.get_loc(v[0]) - b.index.get_loc(c[0])
           for c, v in zip(cor.extrato[::2], cor.extrato[1::2])
           if v.tipo == "mercado"]
print("saídas por tempo: pregões entre compra e venda:",
      set(pregoes))

# carteira comprada e vendida: fração vendida e incerteza
from l03_custos import alvo
print(f"fração vendida média do alvo: "
      f"{-alvo.clip(upper=0).sum(axis=1).mean():.3f}")
for nome, (bruto, giro) in resultado.items():
    anos = len(bruto) / 252
    ep = bruto.std() * np.sqrt(252) / np.sqrt(anos)
    print(f"{nome}: ganho {bruto.mean() * 252:.2%} ± {ep:.2%} a.a.;"
          f" equilíbrio ± {ep / (giro.mean() * 252) * 1e4:.0f} pb")
from l02_custo_ordem import ordem_maxima
from l03_custos import ATIVOS, equilibrio
bd, gd = resultado["diário"]
teto = ordem_maxima(equilibrio(bd, gd)) / (
    gd[gd > 0].mean() / len(ATIVOS))
print(f"capital-teto diário {teto / 1e6:.0f} mi; lucro máximo em "
      f"4/9 dele: {teto * 4 / 9e6:.0f} mi")
print("fechamento 22/02/2021 contra anterior:",
      round(b.Close["2021-02-22"] / b.Close["2021-02-19"] - 1, 4))

# regra de bolso do impacto e exercício novo
from l02_custo_ordem import sigma
print(f"1% do volume: pico {sigma * 0.1 * 1e4:.1f} pb, custo médio "
      f"{2 / 3 * sigma * 0.1 * 1e4:.1f} pb")
for Q in (5e5, 2e6):
    imp = 2 / 3 * 200 * np.sqrt(Q / 50e6)
    tot = 3 + 5 + imp
    print(f"exercício impacto: Q={Q:.0f}: impacto {imp:.1f} pb, "
          f"total {tot:.1f} pb, R$ {Q * tot / 1e4:.0f}")

# imposto e giro (alíquota hipotética de 15%, bruto de 12% a.a.)
fim = 100 * 1.12 ** 10
print(f"imposto só no fim: {fim - 0.15 * (fim - 100):.1f}; "
      f"todo ano: {100 * (1 + 0.12 * 0.85) ** 10:.1f}")
print("aluguel de um mês: R$", round(10000 * 0.02 * 21 / 252, 2))

# Kelly: ponte entre as fórmulas e moeda viciada
print("binária como contínua: mu", 2 * 0.55 - 1, "var",
      1 - 0.10 ** 2, "mu/var", round(0.10 / 0.99, 4))
print("50%:", 1.5 * 0.5, "10%:", round(1.1 * 0.9, 4))
for k in (0.5, 1, 2):
    print(f"g(k f*)/g* com k={k}: {2 * k - k * k}")
