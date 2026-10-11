# Finanças quantitativas com Python

Livro didático do Núcleo de Análise Quantitativa da Liga de Mercado Financeiro (UFU).
Autor: João Emanuel Zilli dos Santos. O PDF pronto é `livro.pdf`.

## O que há nesta pasta

| Caminho | Conteúdo |
|---|---|
| `livro.tex` | arquivo principal |
| `capitulos/` | um `.tex` por capítulo, mais capa, prefácio e apêndices |
| `referencias.tex` | bibliografia (cada capítulo cita só chaves definidas aqui) |
| `lmfquant.sty`, `lmfquant.lua` | projeto gráfico: cores da Liga, fontes, caixas, código |
| `scripts/capNN/` | códigos impressos no livro (`l*.py`), geradores de figuras (`fig_*.py`), conferências e testes (`conf_*.py`, `teste_*.py`, `t_*.py`) e downloads feitos uma vez (`baixar_*.py`) |
| `scripts/estilo.py` | estilo único das figuras |
| `figuras/capNN/` | figuras em PDF, geradas pelos scripts |
| `dados/bruto/` | fontes, nunca editadas: `b3_cotahist.csv` (extrato do histórico oficial da B3), `eventos_societarios.csv` e `correcoes_proventos.csv` (tabelas auditáveis, com a fonte de cada linha) e as cópias do Yahoo Finance (`precos_ajustados.csv`, `precos_sem_ajuste.csv`, `petr4_ohlcv.csv`, `volume.csv`), todas de outubro de 2026 |
| `dados/` | arquivos lidos pelos capítulos, gerados por `scripts/limpar_dados.py`: `precos_sem_ajuste.csv` (fechamento oficial da B3 corrigido por eventos societários), `precos_ajustados.csv` (o mesmo com o ajuste por proventos), `petr4_ohlcv.csv` e `volume_financeiro.csv`; e os `cNN_*.csv` de cada capítulo (séries do Banco Central e resultados intermediários que as figuras leem) |
| `scripts/extrair_cotahist.py`, `limpar_dados.py`, `validar_dados.py`, `conferir_fonte.py`, `baixar_dados.py` | montagem e conferência dos dados (seção "De onde vêm os dados") |
| `fonts/`, `marca/` | fontes livres (STIX Two, Inter, JetBrains Mono) e logotipos |
| `GUIA_DE_ESTILO.md` | regras de escrita, rigor e figuras para quem for alterar o livro |

## Ambiente Python

Python 3.11 ou mais novo (o livro foi gerado com o 3.12) e as versões de `requirements.txt`.
O `Makefile` e `scripts/refazer_figuras.sh` procuram o ambiente virtual em `../.venv`, isto é,
na raiz do repositório, um nível acima desta pasta:

```
cd ..                                  # raiz do repositório
python -m venv .venv
source .venv/bin/activate              # no Windows: .venv\Scripts\activate
pip install -r livro/requirements.txt
```

Quem usa outro Python informa o caminho na variável `PY` (veja abaixo).

## Como rodar um script

Sempre de dentro da pasta do script:

```
cd scripts/cap05
python l01_parametrico.py              # com o ambiente virtual ativado
```

Os scripts leem `../../dados/` e importam módulos da própria pasta (`carteira.py`, `motor.py`)
ou `../estilo.py`. Rodados de outro lugar, falham com `FileNotFoundError` ou
`ModuleNotFoundError`. Os números impressos no livro são exatamente os que eles produzem.
Cada capítulo tem também scripts de conferência, que param com erro se alguma conta do texto
deixar de valer, por exemplo `scripts/cap01/conf_saidas.py` (compara as saídas coladas no
capítulo com a execução) e `scripts/cap01/conf_numeros.py` (refaz os demais números citados).

## Como compilar

Requisitos: LuaLaTeX (TeX Live 2023 ou mais novo, com `luaotfload`, `fontspec`, `unicode-math`,
`tcolorbox`, `tikz`, `listings`, `titlesec`, `imakeidx`).

```
make                              # compila livro.pdf (três passadas)
scripts/compilar_capitulo.sh 05   # compila só um capítulo, em build/cap05/
make limpar                       # apaga auxiliares e compilações isoladas
```

O `make` recompila quando muda um `.tex`, o estilo, uma figura ou um script impresso.
`make limpar` não toca em `build/revisao/`, onde ficam os relatórios das revisões.

No Overleaf: envie a pasta inteira e escolha LuaLaTeX como compilador.

## Como refazer figuras e dados derivados

```
make figuras                              # todos os capítulos (de 5 a 15 minutos)
make figuras PY=/caminho/do/python        # com outro Python
scripts/refazer_figuras.sh 06 08          # só os capítulos indicados
```

Em cada capítulo, o script roda primeiro os geradores, que são os scripts que gravam os
`dados/cNN_*.csv` lidos pelas figuras (hoje, `cap06/l04_duas_janelas.py`,
`cap06/l05_fora_da_amostra.py`, `cap06/l07_estresse_historico.py`, `cap07/sobreajuste.py`,
`cap08/x08_robustez.py` e `cap08/x09_relatorio.py`), e depois todos os `fig_*.py`. Um gerador
novo é encontrado sozinho, desde que grave com `to_csv` e não se chame `baixar_*` nem `fig_*`.
Se algum script falhar, a mensagem aparece e o comando termina com erro. As figuras são
gravadas sem data de criação, de modo que um PDF só muda no Git quando o desenho muda.

Os `baixar_*.py` (CDI e PTAX do Banco Central) dependem da rede, foram rodados uma única vez
e não entram no `make figuras`; o resultado deles está versionado em `dados/`.

## De onde vêm os dados

Os preços das sete ações e do BOVA11 são os fechamentos oficiais da B3. O Ibovespa, o S&P 500
e o dólar vêm do Yahoo Finance, e o CDI e a PTAX, do Banco Central. O ajuste por proventos usa
o fator do Yahoo com as correções de `dados/bruto/correcoes_proventos.csv`. O fluxo é:

```
scripts/extrair_cotahist.py   # uma única vez, com rede: baixa os COTAHIST anuais da B3 para
                              # build/cotahist/ (524 MB, fora do repositório) e grava o extrato
                              # dados/bruto/b3_cotahist.csv; não sobrescreve o que já existe
scripts/baixar_dados.py       # uma única vez, com rede: cópias do Yahoo em dados/bruto/
make dados                    # limpar_dados.py (regrava dados/) e validar_dados.py (confere)
```

`limpar_dados.py` aplica ao fechamento oficial os eventos de `eventos_societarios.csv`, a cisão
da XPart em ITUB4 e, na versão ajustada, o fator de proventos. `validar_dados.py` refaz as contas
por um caminho independente e termina com erro se alguma conferência falhar.
`conferir_fonte.py` compara o Yahoo com a B3 e classifica as divergências; é o módulo que o
Capítulo 2 usa para mostrar por que a fonte foi trocada. Esses scripts podem ser chamados de
qualquer pasta. O extrato e as cópias do Yahoo estão versionados, de modo que `make dados` roda
sem rede e devolve arquivos idênticos.

## Como alterar

1. Leia `GUIA_DE_ESTILO.md` e use `capitulos/cap01.tex` como modelo.
2. Todo número do texto sai de um script em `scripts/capNN/`; toda saída impressa é colada da
   execução real. Ao mudar um script ou os dados, rode de novo e atualize o texto.
3. Não edite os arquivos de fonte em `dados/bruto/`. Um evento societário ou uma correção de
   provento entra como linha nova na tabela correspondente, com a fonte. Para atualizar os dados
   até uma data mais recente, apague de propósito o extrato e as cópias do Yahoo, rode
   `scripts/extrair_cotahist.py`, `scripts/baixar_dados.py` e `make dados`, depois os
   `baixar_*.py` dos capítulos e `make figuras`, e por fim todos os scripts dos capítulos: os
   números do livro vão mudar e precisam ser atualizados no texto.
4. Compile o capítulo, olhe o PDF e só então compile o livro inteiro.

Material educacional. Nada aqui é recomendação de investimento.
