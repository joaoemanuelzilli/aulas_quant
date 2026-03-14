# Aula 4 — Otimização de Portfólio (Fronteira Eficiente de Markowitz)

## Objetivos
- Calcular retorno e risco esperados de portfólios
- Simular portfólios aleatórios (Monte Carlo)
- Identificar o portfólio de máximo Índice de Sharpe e o de mínima variância
- Construir a Fronteira Eficiente via programação quadrática (`cvxpy`)

## Arquivo principal
`aula_04.py` — execute com:
```bash
python aula_04.py
```

## Conceitos cobertos
| Conceito | Descrição |
|----------|-----------|
| `w @ mu` | Retorno esperado do portfólio |
| `w @ Σ @ w` | Variância do portfólio |
| Índice de Sharpe | `(E[R] − Rf) / σ` |
| Fronteira Eficiente | Portfólios de menor risco para cada nível de retorno |
| `cvxpy` | Biblioteca de otimização convexa |
