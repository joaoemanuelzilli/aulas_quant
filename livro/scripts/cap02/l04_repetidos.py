import sys

import pandas as pd

sys.path.insert(0, "..")                 # livro/scripts/
from limpar_dados import ler, sem_negocio

precos = ler("precos_ajustados.csv")     # arquivos brutos
volume = ler("volume.csv")

falso = sem_negocio(precos, volume)      # volume 0 e preço igual
feriado = falso.all(axis=1)              # nenhum papel negociou
tabela = pd.DataFrame({
    "pontos repetidos": falso.sum(),
    "em 2011": falso.loc["2011"].sum(),
    "em dia sem pregão": falso[feriado].sum(),
})
tabela.index = tabela.index.str.replace(".SA", "")
print(tabela)
print(f"\ndatas sem pregão que a fonte preencheu: "
      f"{feriado.sum()}, de {precos.index[feriado][0]:%d/%m/%Y}"
      f" a {precos.index[feriado][-1]:%d/%m/%Y}")
