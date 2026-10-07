from projeto import *

J, K = 12, 2     # fixados em l02_selecao.py, só com o treino
px, ret, cdi = carregar()
carteiras = {"momentum": alvo_momentum(px, J, K),
             "pesos iguais": alvo_fixo(px, ACOES),
             "BOVA11": alvo_fixo(px, ["BOVA11.SA"])}

for nome, periodo in [("TREINO", TREINO), ("TESTE", TESTE)]:
    tabela, serie = {}, {}
    for c, alvo in carteiras.items():
        r, pos, giro = simular(alvo, ret, periodo)
        m = metricas(r, cdi)
        ops = operacoes(pos, px)
        m["giro anual"] = giro.sum() / (len(r) / 252)
        m["operações"] = len(ops)
        m["taxa de acerto"] = (ops["retorno"] > 0).mean()
        tabela[c], serie[c] = m, r
    print(nome, periodo[0][:4], "a", periodo[1][:4])
    print(pd.DataFrame(tabela).round(3).to_string())
    a, ri, t = ativo_mensal(serie["momentum"],
                            serie["pesos iguais"])
    print(f"ativo contra pesos iguais: {a.mean():.2%} ao mês, "
          f"RI = {ri:.2f}, t = {t:.2f}, {len(a)} meses")
    n = len(r)
    taxa = (1 + cdi.loc[r.index]).prod() ** (252 / n) - 1
    sr = tabela["momentum"]["Sharpe"]
    ep = np.sqrt((1 + sr**2 / 504) * 252 / n)   # Lo (2002), iid
    print(f"CDI: {taxa:.2%} a.a.; erro-padrão do Sharpe do "
          f"momentum: {ep:.2f}; pregões: {n}\n")
