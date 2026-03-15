"""
Aula 1 - Parte 1
Introdução ao NumPy com exemplos comentados.
"""

import numpy as np


print("=" * 70)
print("1) CRIAÇÃO DE ARRAYS")
print("=" * 70)

# Criando um array 1D (vetor)
idades = np.array([18, 21, 35, 42, 27])
print("Array de idades:", idades)

# Criando um array 2D (matriz)
notas = np.array(
    [
        [8.5, 7.0, 9.0],
        [6.0, 7.5, 8.0],
        [9.5, 9.0, 9.0],
    ]
)
print("\nMatriz de notas:\n", notas)

# Mostrando formato (linhas x colunas)
print("Formato da matriz de notas:", notas.shape)


print("\n" + "=" * 70)
print("2) OPERAÇÕES VETORIZADAS")
print("=" * 70)

# Somando 1 ano em todas as idades de uma vez
idades_ano_que_vem = idades + 2
print("Idades no ano que vem:", idades_ano_que_vem)

# Aplicando desconto de 10% em um vetor de preços
precos = np.array([100.0, 250.0, 80.0, 40.0, 150.0])
precos_com_desconto = precos * 0.9
print("Preços originais:", precos)
print("Preços com desconto de 10%:", np.round(precos_com_desconto, 2))


print("\n" + "=" * 70)
print("3) INDEXAÇÃO E SLICING")
print("=" * 70)
idades = np.array([18, 21, 35, 42, 27])

#Acessando um elemento específico (índice começa em 0)
print("Primeira idade:", idades[0])

# Pegando um intervalo de elementos
print("Do segundo ao quarto elemento:", idades[1:4])

notas = np.array(
    [
        [8.5, 7.0, 9.0],
        [6.0, 7.5, 8.0],
        [9.5, 9.0, 9.0],
    ])

# Na matriz, podemos acessar linha e coluna
print("Nota da 2ª pessoa na 3ª prova:", notas[1, 2])

# Pegando a primeira coluna inteira
print("Primeira prova (todas as pessoas):", notas[:, 0])


print("\n" + "=" * 70)
print("4) ESTATÍSTICAS RÁPIDAS")
print("=" * 70)

# Cálculos estatísticos em uma linha
print("Média das idades:", np.mean(idades))
print("Maior idade:", np.max(idades))
print("Menor idade:", np.min(idades))
print("Desvio padrão das idades:", round(np.std(idades), 2))

# Média por coluna da matriz de notas
print("Média de cada prova:", np.round(np.mean(notas, axis=0), 2))


print("\n" + "=" * 70)
print("5) EXEMPLO REAL SIMPLES")
print("=" * 70)

# Exemplo: vendas por dia em uma semana
vendas_semana = np.array([1200, 1350, 980, 1500, 1700, 1600, 1900])

# Crescimento percentual do primeiro para o último dia
crescimento = (vendas_semana[-1] - vendas_semana[0]) / vendas_semana[0] * 100

print("Vendas da semana:", vendas_semana)
print("Média diária de vendas:", round(np.mean(vendas_semana), 2))
print("Dia de maior venda:", np.argmax(vendas_semana) + 1)  # +1 para virar 'dia humano'
print("Crescimento do dia 1 para dia 7 (%):", crescimento)
    

print("\nFim da parte de NumPy ✅")
