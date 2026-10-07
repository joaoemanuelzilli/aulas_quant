import pandas as pd


def ler(pasta, nome="precos_ajustados.csv"):
    return pd.read_csv(f"../../dados/{pasta}{nome}",
                       index_col=0, parse_dates=True)


bruto, limpo = ler("bruto/"), ler("")
acoes = ["PETR4.SA", "ITUB4.SA", "BOVA11.SA", "^BVSP"]
print(pd.DataFrame({
    "ausentes no bruto": bruto[acoes].isna().sum(),
    "no limpo": limpo[acoes].isna().sum()}))

dia = "2021-10-04"
for nome, tabela in [("bruto", bruto), ("limpo", limpo)]:
    r = tabela["ITUB4.SA"].pct_change(fill_method=None)
    print(f"ITUB4 em 04/10/2021, {nome}: {r[dia]:+.1%}")

barras = [len(ler(pasta, "petr4_ohlcv.csv"))
          for pasta in ("bruto/", "")]
print("barras de PETR4, bruto e limpo:", barras)
