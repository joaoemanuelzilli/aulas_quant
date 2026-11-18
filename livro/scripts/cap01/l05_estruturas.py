# lista: sequência em ordem, lida pela posição
petr = [26.03, 26.84, 26.62, 26.68]  # PETR4, 2 a 5/1/2024
petr.append(26.48)                   # 8/1/2024, vai para o fim
print(petr[0], petr[-1], len(petr))

# tupla: como a lista, mas não pode ser alterada
dia = ("2024-01-02", 26.03)
data, preco = dia                    # desempacota os dois campos
try:
    dia[1] = 27.0
except TypeError:
    print("a tupla não aceita atribuição")

# dicionário: cada valor é consultado pela sua chave
setor = {"PETR4": "petróleo", "VALE3": "mineração",
         "ITUB4": "banco", "BBAS3": "banco"}
setor["WEGE3"] = "equipamentos"      # acrescenta um par
print(setor["ITUB4"], "ABEV3" in setor)

# conjunto: sem repetição, feito para testar pertinência
setores = set(setor.values())
print(len(setores), sorted(setores), "banco" in setores)
