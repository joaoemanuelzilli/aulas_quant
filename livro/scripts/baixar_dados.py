"""Baixa do Yahoo Finance a cópia bruta de índices, câmbio e ações.

Roda uma única vez; os arquivos ficam em livro/dados/bruto/ e não são
sobrescritos. Os preços das ações usados no livro vêm da B3 (ver
extrair_cotahist.py); o Yahoo fornece índices, câmbio e o fator de
proventos. A cópia do livro foi baixada em 6 de outubro de 2026.
"""
from pathlib import Path

import yfinance as yf

BRUTO = Path(__file__).resolve().parent.parent / "dados" / "bruto"
TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "ABEV3.SA",
           "BBAS3.SA", "MGLU3.SA", "BOVA11.SA", "^BVSP", "^GSPC", "BRL=X"]
PERIODO = dict(start="2010-01-01", end="2026-01-01", progress=False)

if (BRUTO / "precos_ajustados.csv").exists():
    raise SystemExit("dados/bruto/ já existe; nada foi baixado.")
BRUTO.mkdir(parents=True, exist_ok=True)

ajustado = yf.download(TICKERS, auto_adjust=True, **PERIODO)
ajustado["Close"].to_csv(BRUTO / "precos_ajustados.csv")
ajustado["Volume"].filter(like=".SA").to_csv(BRUTO/"volume.csv")

sem_ajuste = yf.download(TICKERS, auto_adjust=False, **PERIODO)
sem_ajuste["Close"].to_csv(BRUTO / "precos_sem_ajuste.csv")

ohlc = yf.download("PETR4.SA", auto_adjust=True, **PERIODO)
ohlc.columns = ohlc.columns.get_level_values(0)
ohlc.to_csv(BRUTO / "petr4_ohlcv.csv")
