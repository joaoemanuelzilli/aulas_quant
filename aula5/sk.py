import yfinance as yf

ticker = "PETR4.SA"
df = yf.download(ticker, start="2023-01-01", end="2024-01-01", auto_adjust=True)

precos = df['Close'].values.flatten()
retornos = []
for i in range(1, len(precos)):
    if precos[i-1] != 0:
        retornos.append((precos[i] / precos[i-1]) - 1)

def calcular_assimetria(dados):
    n = len(dados)
    if n < 3: return 0
    
    media = sum(dados) / n
    
    soma_quadrados = 0      
    soma_cubos = 0          
    
    for x in dados:
        desvio = x - media
        soma_quadrados += desvio**2
        soma_cubos += desvio**3
        
    m2 = soma_quadrados / n
    m3 = soma_cubos / n
    
    if m2 == 0: return 0
    
    assimetria = m3 / (m2**(1.5))
    return assimetria

# 3. Execução
resultado_skew = calcular_assimetria(retornos)

print(f"\n--- Análise de Assimetria: {ticker} ---")
print(f"Assimetria Calculada: {resultado_skew:.4f}")


if resultado_skew > 0:
    print("Interpretação: Assimetria Positiva.") #grandes perdas são comuns nesse tipo de cauda 
elif resultado_skew < 0:
    print("Interpretação: Assimetria Negativa.")
else:
    print("Interpretação: Simétrica.")