import numpy as np
import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
p = precos[["^BVSP", "^GSPC"]]
p = p.dropna(subset="^BVSP")          # só os dias de pregão na B3

sp = p["^GSPC"]
opcoes = pd.DataFrame({
    "original": sp,
    "ffill": sp.ffill(),              # repete o último valor
    "interpolate": sp.interpolate(),  # reta entre os vizinhos
    "bfill": sp.bfill(),              # copia o valor seguinte
})
print(opcoes.loc["2024-11-26":"2024-12-02"].round(2))


def resumo(tabela):
    r = tabela.pct_change().dropna()
    return (len(r), r["^GSPC"].std() * np.sqrt(252),
            (r["^GSPC"] == 0).sum(), r.corr().iloc[0, 1])


casos = {"dropna": p.dropna(), "ffill": p.ffill(),
         "interpolate": p.interpolate()}
print("\n             dias   vol S&P  zeros  correlação")
for nome, tabela in casos.items():
    n, vol, zeros, corr = resumo(tabela)
    print(f"{nome:<12}{n:>5}{vol:>10.2%}{zeros:>7}{corr:>12.4f}")

# se uma fração f dos retornos vira zero, a vol cai por raiz(1-f)
vol_real, n_real = resumo(casos["dropna"])[1], len(p.dropna()) - 1
f = (len(p) - 1 - n_real) / (len(p) - 1)
print(f"\nf = {f:.4f}   vol prevista com ffill: "
      f"{vol_real * np.sqrt(1 - f):.2%}")
