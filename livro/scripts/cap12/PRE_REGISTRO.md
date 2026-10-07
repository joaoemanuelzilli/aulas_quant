# Ficha de pré-registro: momentum entre sete ações da B3

Reconstituição didática. Esta ficha reproduz a Tabela 12.3 do
livro e mostra o formato do documento. Ela NÃO tem o valor de
um pré-registro de verdade, por dois motivos declarados no
Capítulo 12: (1) o Capítulo 8 já examina momentum transversal
nestas mesmas sete ações com dados até 2025, de modo que o
período de teste já era conhecido quando a ficha foi escrita;
(2) a pasta do livro não tem um commit datado anterior à
rodada de teste. No projeto do seu grupo, grave a ficha em um
commit antes de rodar qualquer coisa no período de teste.

## Afirmação
Entre sete ações da B3, uma carteira das K de maior retorno
nos últimos J meses supera, no mês seguinte, a carteira de
pesos iguais das sete, líquida de custos.

## Mecanismo
Reação lenta dos preços a informação específica das empresas
(Jegadeesh e Titman, 1993).

## Universo e horizonte
PETR4, VALE3, ITUB4, WEGE3, ABEV3, BBAS3 e MGLU3. Somente
posições compradas. Rebalanceamento mensal.

## Períodos
Aquecimento: 2010 a 2012. Treino: 2013 a 2019.
Teste: 2020 a 2025.

## Variantes
Doze: J em {3, 6, 9, 12} e K em {2, 3, 4}. Escolhe-se a de
maior Sharpe líquido no treino.

## Execução e custos
Sinal com o fechamento do último pregão do mês. Ordem no
fechamento do pregão seguinte (um pregão de atraso em relação
à convenção padrão do livro, a execução no fechamento do
sinal). Custo de 0,10% do valor negociado, por ponta.

## Benchmarks
Principal: pesos iguais nas sete ações, com o mesmo
rebalanceamento e os mesmos custos.
Secundário: BOVA11 comprado e mantido.

## Métricas
As obrigatórias do enunciado, mais a razão de informação e a
estatística t do retorno ativo mensal contra o benchmark
principal.

## Regra de decisão
Confirmada se, no teste, o Sharpe supera o do benchmark
principal e t >= 2. Refutada se o retorno ativo médio é nulo
ou negativo. Inconclusiva em qualquer outro caso.
