# Aula 5 — Gestão de Risco: VaR e CVaR

## Objetivos
- Calcular o VaR histórico e paramétrico de um ativo
- Calcular o CVaR (Expected Shortfall)
- Realizar backtesting do VaR com janela móvel
- Interpretar resultados e identificar falhas do modelo

## Arquivo principal
`aula_05.py` — execute com:
```bash
python aula_05.py
```

## Conceitos cobertos
| Conceito | Descrição |
|----------|-----------|
| VaR Histórico | Quantil empírico da distribuição de perdas |
| VaR Paramétrico | Baseado em média e volatilidade (distribuição normal) |
| CVaR / ES | Média das perdas além do VaR (cauda da distribuição) |
| Backtesting | Conta violações do VaR num período fora da amostra |
