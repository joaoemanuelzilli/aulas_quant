# Momentum entre sete ações da B3

Projeto-exemplo do Capítulo 12 do livro do Núcleo Quant (LMF,
UFU). No projeto do seu grupo, esta linha traz os nomes e os
cursos dos integrantes.

## Tese
Entre sete ações líquidas da B3, as duas de maior retorno nos
últimos doze meses rendem mais, no mês seguinte, do que a
carteira de pesos iguais das sete, líquido de custos.

## Como rodar (Python 3.12)
    pip install -r requirements.txt
    python l01_sanidade.py
    python l02_selecao.py
    python l03_teste.py
    python l04_robustez.py
    python l05_operacoes.py

Os dados ficam congelados em ../../dados (preços do livro e
c12_cdi.csv, gravado uma vez por baixar_cdi.py).

## Conclusão
Inconclusivo. No teste (2020 a 2025): Sharpe 0,45 contra 0,14
do benchmark, t do retorno ativo = 1,08. O resultado depende
de 2020 e da presença de MGLU3 no universo, e o universo já
tinha sido examinado no Capítulo 8 com dados até 2025.
