from calculo import derivada, preco_ltn

taxa, du = 0.1436, 532   # exemplo da metodologia do Tesouro
P = preco_ltn(taxa, du)
exata = -(du / 252) * P / (1 + taxa)   # P'(taxa), em R$
print(f"preço: R$ {P:.6f}   P'(taxa) exata: {exata:.4f}")

f = lambda y: preco_ltn(y, du)
for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    fr = derivada(f, taxa, h, "frente")
    ce = derivada(f, taxa, h, "central")
    print(f"h = {h:.0e}   erro frente {fr - exata:+.1e}"
          f"   erro central {ce - exata:+.1e}")
print(f"queda por 1 pb de alta: R$ {-exata * 1e-4:.4f}")
