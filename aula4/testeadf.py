from statsmodels.tsa.stattools import adfuller

series = [12, 15, 14, 18, 17, 20, 22, 21, 23, 25, 24, 26, 28, 27, 30, 33, 23, 29, 28, 30]
# series2 = [12, 15, 14, 18, 11, 13, 15, 19, 11, 14, 16, 20, 12, 17, 19, 12, 13, 18, 20, 12]

result = adfuller(series)

# result[0]   estatística ADF
# result[1]   p-value
# result[2]   número de lags usados
# result[3]   número de observações
# result[4]   valores críticos
# result[5]   informação (AIC ou similar)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("A serie e estacionária.")    
else:    print("A serie nao eé estacionaria.")   

media = sum(series) / len(series)
print("Media:", media)

# Calcular a média da primeira metade
primeira_metade = series[:len(series)//2]
media_primeira = sum(primeira_metade) / len(primeira_metade)
print("Media da primeira metade:", media_primeira)

# Calcular a média da segunda metade
segunda_metade = series[len(series)//2:]
media_segunda = sum(segunda_metade) / len(segunda_metade)
print("Media da segunda metade:", media_segunda)
