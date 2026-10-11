import pandas as pd

precos = pd.read_csv("../../dados/bruto/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
acoes = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA",
         "ABEV3.SA", "WEGE3.SA"]
p = precos[acoes + ["^BVSP"]].dropna()
r = p.pct_change().dropna()

# retorno da ação menos o do mercado no mesmo dia
excesso = r[acoes].sub(r["^BVSP"], axis=0)

# stack() empilha as colunas: uma linha por (data, ação)
longo = pd.DataFrame({
    "excesso": excesso.stack(),
    "retorno": r[acoes].stack(),
    "Ibovespa": (r[acoes] - excesso).stack(),
    "dia seguinte": r[acoes].shift(-1).stack(),
})
suspeitos = longo[longo["excesso"].abs() > 0.15]
print(suspeitos.map(lambda v: f"{v:+.1%}"))
