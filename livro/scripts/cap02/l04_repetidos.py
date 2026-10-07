import sys

import pandas as pd

sys.path.insert(0, "..")                 # livro/scripts/
from limpar_dados import ler, sem_negocio

tela = ler("precos_sem_ajuste.csv")      # arquivos brutos
ajustado = ler("precos_ajustados.csv")
volume = ler("volume.csv")
acoes = list(volume.columns)

falso = pd.DataFrame(
    {a: sem_negocio(tela[a], volume[a]) for a in acoes})
feriado = falso.all(axis=1)              # nenhum papel negociou
# a regra ingênua: preço ajustado igual ao da linha anterior
ingenua = (ajustado[acoes].diff() == 0) & (volume == 0)

tabela = pd.DataFrame({
    "repetidos": falso.sum(),
    "em 2011": falso.loc["2011"].sum(),
    "regra ingênua": ingenua.sum(),
})
tabela.index = tabela.index.str.replace(".SA", "")
print(tabela)
print(f"\ndatas sem pregão que a fonte preencheu: "
      f"{feriado.sum()}, de {tela.index[feriado][0]:%d/%m/%Y}"
      f" a {tela.index[feriado][-1]:%d/%m/%Y}")

exemplo = pd.DataFrame({"sem ajuste": tela["PETR4.SA"],
                        "ajustado": ajustado["PETR4.SA"],
                        "volume": volume["PETR4.SA"]})
print(exemplo.loc["2011-05-10":"2011-05-13"].round(3))
