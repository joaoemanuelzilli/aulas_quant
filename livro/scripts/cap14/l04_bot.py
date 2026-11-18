import pandas as pd
from papel import CorretoraPapel

b = pd.read_csv("../../dados/petr4_ohlcv.csv", index_col=0,
                parse_dates=True).loc["2018":"2024"]
media = b.Close.rolling(100).mean()

RISCO, STOP, ALVO, RECUO, PRAZO = 0.01, 0.05, 0.08, 0.02, 10
cor = CorretoraPapel(caixa=100_000)
curva, dias = {}, 0

for dia, barra in b.loc["2019":].iterrows():
    cor.processar(dia, barra)          # 1. execuções da barra
    abertas, c = cor.abertas(), barra.Close
    if cor.posicao == 0:               # 2. decisão no fechamento
        for o in abertas:              # entrada ou perna órfã
            cor.cancelar(o)
        if c > media[dia]:             # só em tendência de alta
            limite = round(c * (1 - RECUO), 2)
            risco = RISCO * cor.patrimonio(c)
            qtd = int(risco / (STOP * limite) / 100) * 100
            cor.enviar("compra", qtd, "limitada", limite)
        dias = 0
    elif not abertas:                  # comprou: protege
        pe = cor.extrato[-1].preco     # preço de entrada
        q = cor.posicao
        cor.enviar("venda", q, "stop", round(pe * (1 - STOP), 2))
        alvo = round(pe * (1 + ALVO), 2)
        cor.enviar("venda", q, "limitada", alvo)
    else:
        dias += 1
        if dias == PRAZO:              # saída por tempo
            for o in abertas:
                cor.cancelar(o)
            cor.enviar("venda", cor.posicao)
    curva[dia] = cor.patrimonio(c)     # 3. registro

curva = pd.Series(curva)
if __name__ == "__main__":
    print("data       ordem lado   tipo      qtd   preço  taxa"
          "  pos.     caixa")
    for d, i, lado, tipo, q, p, taxa, pos, cx in cor.extrato[:8]:
        print(f"{d:%Y-%m-%d} {i:<5} {lado:6} {tipo:8} {q:4d}"
              f" {p:7.4f} {taxa:5.2f} {pos:5d} {cx:9.2f}")
    estados = pd.Series([o.estado for o in cor.ordens])
    print(estados.value_counts().to_string())
    feitas = [o for o in cor.ordens if o.estado == "executada"]
    for lado, t in (("compra", "limitada"), ("venda", "stop"),
                    ("venda", "limitada"), ("venda", "mercado")):
        n = sum((o.lado, o.tipo) == (lado, t) for o in feitas)
        m = sum((o.lado, o.tipo) == (lado, t) for o in cor.ordens)
        print(f"{lado} {t}: {n} executadas de {m} enviadas")
    pior = min((o for o in feitas if o.tipo == "stop"),
               key=lambda o: o.preco_exec / o.preco)
    quando = [e[0] for e in cor.extrato if e[1] == pior.id][0]
    print(f"pior stop: {quando:%Y-%m-%d}, gatilho "
          f"{pior.preco:.2f}, executado a {pior.preco_exec:.4f}")
    taxas = sum(e[6] for e in cor.extrato)
    print(f"patrimônio final: {curva.iloc[-1]:,.2f}"
          f"   taxas pagas: {taxas:,.2f}")
    queda = (curva / curva.cummax() - 1).min()
    manter = b.Close.loc["2019":]
    print(f"drawdown máximo: {queda:.1%}   comprar e "
          f"manter: {manter.iloc[-1] / manter.iloc[0] - 1:+.0%}")
