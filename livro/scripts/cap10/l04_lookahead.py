import metricas as m
from motor import carregar, cruzamento

preco, ret = carregar("PETR4.SA")
r, rf = ret["ativo"], ret["cdi"]


def regra_rapida(defasagem):
    """Comprado se o fechamento supera a média de 5 dias."""
    sinal = (preco > preco.rolling(5).mean()).astype(float)
    posicao = sinal.shift(defasagem).fillna(0.0)
    giro = posicao.diff().abs().fillna(0.0)
    return posicao * r + (1 - posicao) * rf - 0.001 * giro


def regra_lenta(defasagem):
    return cruzamento(preco, rf, defasagem=defasagem)["retorno"]


livre = rf.loc["2011":"2025"]
for regra in [regra_rapida, regra_lenta]:
    for defasagem in [0, 1]:
        liquido = regra(defasagem).loc["2011":"2025"]
        print(f"{regra.__name__}, shift({defasagem}): "
              f"retorno a.a. {m.retorno_anual(liquido):6.1%}   "
              f"Sharpe {m.sharpe(liquido, livre):5.2f}")

# por que a regra rápida "funciona" sem o shift
acima = preco > preco.rolling(5).mean()
trocas = acima.astype(float).diff().abs().loc["2011":"2025"]
acima, r = acima.loc["2011":"2025"], r.loc["2011":"2025"]
print(f"trocas de posição por ano: {trocas.sum() / 15:.0f}")
print(f"dias de alta com o sinal do próprio dia ligado: "
      f"{(r[acima] > 0).mean():.1%}; desligado: "
      f"{(r[~acima] > 0).mean():.1%}")
