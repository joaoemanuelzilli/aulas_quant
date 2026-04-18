import yfinance as yf

ticker = "PETR4.SA"
df = yf.download(ticker, start="2023-01-01", end="2024-01-01", auto_adjust=True)

if 'Close' in df.columns:
    precos = df['Close'].values.flatten() 
else:
    precos = df.columns.get_level_values(0) 
    precos = df.iloc[:, df.columns.get_level_values(1) == 'Close'].values.flatten()



if len(precos) == 0:
    print("Erro")
else:
    retornos = []
    for i in range(1, len(precos)):
        if precos[i-1] != 0:
            retorno = (precos[i] / precos[i-1]) - 1
            retornos.append(retorno)

    # --- CÁLCULO MANUAL DA CURTOSE ---
    def calcular_curtose(dados):
        n = len(dados)
        if n < 4: return 0
        media = sum(dados)/n
        
        '''
        media=0
        for e in dados:
            media+= e
        media=(media/n)
        '''
        
        soma_quadrados = 0      
        soma_quarta_potencia = 0 
        
        for x in dados:
            desvio = x - media
            soma_quadrados += desvio**2
            soma_quarta_potencia += desvio**4
            
        m2 = soma_quadrados / n
        m4 = soma_quarta_potencia / n
        
        # Se a variância (m2) for zero, a curtose é indefinida
        if m2 == 0: return 0
        
        return (m4 / (m2**2)) - 3

    resultado = calcular_curtose(retornos)

    print(f"\n--- Resultados para {ticker} ---")
    print(f"Número de pregões analisados: {len(retornos)}")
    print(f"Excesso de Curtose: {resultado:.4f}")

    if resultado > 0:
        print("Interpretação: (Caudas longas/Risco de cauda).")
    elif resultado < 0:
        print("Interpretação: (Caudas curtas).")
    else:
        print("Interpretação: (Distribuição Normal).")