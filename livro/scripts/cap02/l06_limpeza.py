import pandas as pd


def ler(pasta, nome="precos_ajustados.csv"):
    return pd.read_csv(f"../../dados/{pasta}{nome}",
                       index_col=0, parse_dates=True)


bruto, limpo = ler("bruto/"), ler("")
ativos = ["PETR4.SA", "ITUB4.SA", "BOVA11.SA", "^BVSP"]
print(pd.DataFrame({
    "ausentes no bruto": bruto[ativos].isna().sum(),
    "no limpo": limpo[ativos].isna().sum()}))

dia = "2021-10-04"
for nome, tabela in [("bruto", bruto), ("limpo", limpo)]:
    r = tabela["ITUB4.SA"].pct_change(fill_method=None)
    print(f"ITUB4 em 04/10/2021, {nome}: {r[dia]:+.1%}")

barras = [len(ler(pasta, "petr4_ohlcv.csv"))
          for pasta in ("bruto/", "")]
print("barras de PETR4, bruto e limpo:", barras)

# depois de dropna(), quantos retornos cobrem mais de um pregão?
acoes = [c for c in limpo.columns if c.endswith(".SA")]
pregoes = limpo.index[limpo[acoes].notna().any(axis=1)]
ordem = pd.Series(range(len(pregoes)), index=pregoes)
linhas = {}
for a in acoes:
    preco = limpo[a].dropna()
    passo = ordem[preco.index].diff()    # pregões entre preços
    linhas[a.replace(".SA", "")] = {
        "retornos longos": (passo > 1).sum(),
        "maior intervalo": int(passo.max())}
print("\n", pd.DataFrame(linhas).T, sep="")

petr = limpo["PETR4.SA"]
sem_linha = petr.dropna().pct_change()
no_lugar = petr.pct_change(fill_method=None)
print(f"\nPETR4 em 17/08/2011, depois de dropna(): "
      f"{sem_linha['2011-08-17']:+.1%}")
print("com os ausentes no lugar:", no_lugar["2011-08-17"])
