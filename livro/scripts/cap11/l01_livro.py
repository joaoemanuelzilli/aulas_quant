# livro de exemplo: (preço, quantidade), melhor oferta primeiro
compras = [(20.00, 800), (19.99, 1500), (19.98, 2200),
           (19.97, 1200), (19.96, 3000)]
vendas = [(20.02, 500), (20.03, 1200), (20.04, 1800),
          (20.05, 900), (20.06, 2500)]


def varrer(lado_oposto, qtd):
    """Executa uma ordem a mercado contra o livro; preço médio."""
    falta, gasto = qtd, 0.0
    for preco, disponivel in lado_oposto:
        q = min(falta, disponivel)
        gasto += q * preco
        falta -= q
        if falta == 0:
            return gasto / qtd
    raise ValueError("livro sem profundidade para a ordem")


if __name__ == "__main__":
    bid, ask = compras[0][0], vendas[0][0]
    meio = (bid + ask) / 2
    print(f"spread: {ask - bid:.2f} = "
          f"{(ask - bid) / meio * 1e4:.1f} pb do preço médio "
          f"{meio:.2f}")
    for qtd in (100, 500, 2000, 4000):
        pm = varrer(vendas, qtd)
        print(f"compra de {qtd:4d}: preço médio {pm:.4f}, "
              f"{(pm / meio - 1) * 1e4:4.1f} pb acima do meio")
