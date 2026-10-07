from carteira import ACOES, retornos
from experimento import carregar_cdi, resumo, simular

r = retornos("2012", "2025")
cdi = carregar_cdi(r.index)
diario, giro = simular(r, cdi)
diario.to_csv("../../dados/c06_fora_da_amostra.csv")
print(f"{diario.index[0]:%d/%m/%Y} a {diario.index[-1]:%d/%m/%Y}")
print(resumo(diario, giro, cdi).round(2).to_string())

print("\nsem MGLU3:")
seis = [a for a in ACOES if a != "MGLU3.SA"]
diario6, giro6 = simular(r[seis], cdi)
diario6.to_csv("../../dados/c06_fora_da_amostra_sem_mglu.csv")
print(resumo(diario6, giro6, cdi).round(2).to_string())
