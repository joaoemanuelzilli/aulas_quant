# Guia de estilo do livro do Núcleo Quant

Vale para quem escreve ou revisa capítulos. O capítulo-modelo é `capitulos/cap01.tex`:
leia-o inteiro antes de escrever, junto com `lmfquant.sty` (macros) e `scripts/estilo.py` (figuras).

## Leitor e voz

- Leitor: estudante de graduação (engenharias, computação, economia) que sabe programar um pouco
  e viu estatística básica. Inteligente, sem experiência de mercado.
- Português do Brasil, direto, na segunda pessoa ("você"). Frases completas, parágrafos que
  explicam o porquê antes da fórmula. Primeiro a ideia e um exemplo com números, depois a definição.
- Termos de mercado em inglês ficam em inglês quando é assim que se fala na mesa
  (`\ingles{drawdown}`, `\ingles{backtest}`), sempre explicados na primeira ocorrência.
- Proibido (soa artificial e não acrescenta nada): "vale ressaltar", "é importante notar",
  "em resumo", "em suma", "mergulhar", "jornada", "no mundo das finanças", "crucial", "robusto" como
  elogio vazio, "poderoso", perguntas retóricas em série, construções "não é X, é Y" repetidas,
  listas de três adjetivos, travessões no meio da frase, emojis, frases de efeito no fim de parágrafo,
  abrir seção anunciando o que a seção vai fazer, fechar seção repetindo o que ela fez.
- Sem negrito espalhado. Negrito só via `\termo{}` (primeira ocorrência de um conceito, entra no
  índice) e no início de itens de lista quando o item tem um nome.
- Decimal com vírgula no texto e nas fórmulas (`$0,5$` funciona direto; para separar argumentos
  use vírgula seguida de espaço: `$f(x, y)$`). Saídas de código ficam como o Python imprime.
- Convenções: 252 dias úteis; nível de confiança do VaR escrito como 95% ou 99%; taxa livre de
  risco brasileira = CDI; retornos simples `R`, log-retornos `r`; vetores em negrito `\vetor{w}`,
  matriz de covariância `\symbf{\Sigma}`, transposta `\T`, `\E`, `\Var`, `\Cov`, `\Corr`, `\VaR`, `\CVaR`.

## Rigor (o ponto mais importante)

- Toda afirmação técnica precisa estar certa e, quando não for consenso, atribuída a uma fonte
  com `\citet{}`/`\citep{}`. Só cite o que você tem certeza de que a fonte diz. Não invente número de
  página, capítulo, citação literal ou resultado empírico. Na dúvida, afirme menos.
- Chaves disponíveis: as de `referencias.tex`. Se precisar de outra referência, NÃO edite
  `referencias.tex`: crie `capitulos/capNN_refs.tex` com os `\bibitem[Autor(Ano)]{chave}` completos
  no mesmo formato, e só inclua obras que você tem certeza de que existem com aqueles dados.
- Todo número no texto, em tabela ou em legenda sai de um script do capítulo. Nada de número
  "ilustrativo" inventado em análise de dados reais. Exemplos didáticos feitos à mão (contas de
  duas linhas) devem ser conferidos.
- Todo código impresso é um arquivo testado em `scripts/capNN/` e entra com `\lstinputlisting`
  (pode usar `firstline`/`lastline`). A saída mostrada (`style=saida`) é colada da execução real.
  Linhas de código com no máximo 66 caracteres (a coluna é estreita). Código curto e legível:
  é material de ensino, não biblioteca.
- Código que depende de serviço externo (corretora, API paga) pode aparecer como esboço em
  `\begin{lstlisting}[style=py]`, com o texto dizendo claramente que é um esboço não executado.
- Dados: use o conjunto congelado em `dados/` (ver `scripts/estilo.py`, função `precos`):
  `precos_ajustados.csv` e `precos_sem_ajuste.csv` com PETR4.SA, VALE3.SA, ITUB4.SA, WEGE3.SA,
  ABEV3.SA, BBAS3.SA, MGLU3.SA, BOVA11.SA, ^BVSP, ^GSPC, BRL=X, diários de 2010 a 2025, e
  `petr4_ohlcv.csv`. Se precisar de outra série (CDI, por exemplo), baixe uma vez com um script
  `scripts/capNN/baixar_*.py`, grave em `dados/` e leia do arquivo. Sementes aleatórias fixas.
- Python do projeto: `/home/joaoemanuel/git/Quant/.venv/bin/python` (numpy, pandas, scipy,
  statsmodels, scikit-learn, arch, xgboost, lightgbm, backtrader, matplotlib, seaborn, yfinance).

## Estrutura de um capítulo

```
\chapter{Título}  \label{cap:...}
\abertura{um parágrafo: o problema que o capítulo resolve}
\begin{objetivos} \item ... \end{objetivos}        (4 a 6 itens, verbos no infinitivo)
\section{...} ...                                   (6 a 10 seções)
\begin{resumo} \item ... \end{resumo}               (4 a 6 itens)
\begin{exercicios} \item \nivel{básico} ... \end{exercicios}   (6 a 9; níveis: básico, intermediário, desafio)
\begin{leituras} \item \citet{...}, capítulo tal, para ... \end{leituras}
```

Tamanho: 14 a 20 páginas compiladas. Rótulos dos capítulos (use `Capítulo~\ref{...}`):
`cap:retornos` (1), `cap:dados` (2), `cap:estatistica` (3), `cap:series` (4), `cap:risco` (5),
`cap:portfolio` (6), `cap:backtest` (7), `cap:estrategias` (8), `cap:ml1` (9), `cap:ml2` (10),
`cap:trading` (11), `cap:projeto` (12). Prefixe seus rótulos internos com o número do capítulo
(`fig:c05-...`, `eq:c05-...`, `sec:c05-...`, `cod:c05-...`, `tab:c05-...`) para não colidir.

## Blocos disponíveis (definidos em lmfquant.sty)

| Bloco | Uso |
|---|---|
| `definicao{Título}` | definição formal, numerada |
| `resultado{Título}` | proposição ou fórmula que se demonstra ou justifica |
| `intuicao` | a ideia em palavras, antes ou depois da fórmula |
| `armadilha{Título}` | erro comum e como evitá-lo |
| `exemplo{Título}` | exemplo numérico resolvido |
| `repositorio` | aponta para os scripts das aulas (`\arq{aula5/Var_historico.py}`) |

Caixas curtas (até umas 10 linhas); não quebram entre páginas, exceto `exemplo`. Não empilhe
duas caixas seguidas nem transforme o capítulo em sequência de caixas: o texto corrido é o
principal. Uma ou duas `intuicao` e duas a quatro `armadilha` por capítulo é a medida.
Notas de margem: `\nota{...}` (curtas, uma ou duas por página no máximo).
Tabelas com `booktabs` (`\toprule`, `\midrule`, `\bottomrule`), cabeçalho com `\cab{}`, sem linhas
verticais, legenda acima. Conteúdo que precisa da largura total: `\begin{largo}...\end{largo}`.

## Figuras

- Geradas por `scripts/capNN/fig_*.py` usando `estilo.py`; gravadas em `figuras/capNN/` como PDF.
  Entram com `\figura[t]{capNN/nome}{legenda}{fig:cNN-nome}` (largura do texto, 118 mm),
  `\figuralarga` (168 mm) ou `\figuramargem` (43 mm, só para gráficos muito simples).
- 5 a 9 figuras por capítulo. Cada figura mostra uma ideia. A legenda diz o que se deve ver.
- Regras de desenho: cores só de `estilo.py`, na ordem ROXO, OCRE, VERDE, AZUL, ROSA para séries
  (cinza para referência/contexto); sem eixo duplo (use dois painéis); rótulo direto no fim da
  linha em vez de caixa de legenda quando houver até 4 séries; texto dos rótulos em preto ou cinza,
  nunca na cor da série; sem título redundante com a legenda; magnitude em uma cor só
  (`SEQ_ROXO`), valores com sinal em `DIV`; nada de arco-íris, 3D ou gradiente decorativo.
- Diagramas conceituais (fluxos, linhas do tempo de treino/teste) em TikZ, mesmo vocabulário:
  traço fino preto, preenchimento `lilas`, destaque `roxo`, texto `\sffamily\footnotesize`,
  cantos retos.
- Depois de gerar, OLHE cada figura (converta com `pdftoppm -r 110 -png` e abra o PNG) e corrija
  rótulos sobrepostos, texto cortado, eixos ilegíveis.

## Conferência antes de entregar

1. `scripts/compilar_capitulo.sh NN` compila sem erro (gera `build/capNN/capNN.pdf`).
   Referências a outros capítulos aparecem como "??" na compilação isolada; isso é esperado.
2. Sem `Overfull \hbox` acima de 5pt (o script lista).
3. Abra as páginas (`pdftoppm -r 70 -png`) e confira o resultado visual.
4. Releia procurando: número sem script, afirmação sem base, vocabulário proibido, código que
   não é o arquivo testado.
