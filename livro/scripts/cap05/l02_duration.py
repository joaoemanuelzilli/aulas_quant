from calculo import convexidade, duration_mod, preco_ltn

taxa = 0.1436
for du in [532, 2520]:              # ~2,1 anos e 10 anos
    D, C = duration_mod(taxa, du), convexidade(taxa, du)
    P = preco_ltn(taxa, du)
    print(f"du = {du}: D = {D:.4f}  C = {C:.4f}")
    print("choque   exata   1a ordem  2a ordem  (variação %)")
    for pb in [-300, -100, 100, 300]:
        dy = pb / 10_000
        exata = preco_ltn(taxa + dy, du) / P - 1
        ordem1 = -D * dy
        ordem2 = ordem1 + 0.5 * C * dy**2
        print(f"{pb:+5d}  {exata:+8.3%} {ordem1:+8.3%}"
              f"  {ordem2:+8.3%}")
