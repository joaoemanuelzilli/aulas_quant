# Aula 7 — Vieses em Backtesting & Backtrader

## Temas

| Arquivo | Tema |
|---------|------|
| `01_lookahead_bias.py`       | Look-Ahead Bias — usar o futuro sem querer |
| `02_survivorship_bias.py`    | Survivorship Bias — ignorar quem faliu |
| `03_overfitting.py`          | Overfitting — decorar o passado |
| `04_backtrader_intro.py`     | Backtrader: estrutura básica e primeiro backtest |
| `05_backtrader_otimizacao.py`| Backtrader: otimização com separação in/out-of-sample |

## Dependências

```bash
pip install backtrader yfinance matplotlib pandas numpy
```

## Ordem sugerida

1. Rode os três arquivos de vieses (01, 02, 03) para entender **o que pode dar errado**
2. Rode o 04 para aprender a estrutura do Backtrader
3. Rode o 05 para ver como otimizar parâmetros sem cair em overfitting

## Resumo dos Vieses

**Look-Ahead Bias**: a estratégia usa dados que não existiriam na hora da decisão.
→ Solução: `rolling(..., center=False)`, nunca `shift` negativo em sinais.

**Survivorship Bias**: testar apenas ativos que sobreviveram até hoje.
→ Solução: usar bases de dados point-in-time com ativos deslistados.

**Overfitting**: otimizar parâmetros no mesmo período que você avalia.
→ Solução: separar treino/teste, walk-forward, poucos parâmetros.
