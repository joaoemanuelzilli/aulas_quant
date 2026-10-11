"""Extrai dos arquivos da B3 as cotações oficiais do livro.

A B3 publica um arquivo por ano com as cotações diárias de
todos os mercados, o COTAHIST_AAAAA.ZIP (cerca de 520 MB de
2010 a 2025). Cada linha do texto tem 245 caracteres e cada
campo ocupa colunas fixas, descritas no documento "Layout
do arquivo de cotações históricas" da própria bolsa.

Este script baixa os arquivos anuais que ainda não estiverem
na pasta build/cotahist/ (nunca sobrescreve um que já
exista), guarda só as linhas do mercado à vista dos papéis
do livro e grava dados/bruto/b3_cotahist.csv. Esse extrato é
a cópia bruta oficial: entra no repositório e não é editado.
Os ZIP não entram.

Roda uma única vez. Para refazer, apague o extrato de
propósito. A cópia do livro foi extraída em 10/10/2026.
"""
import csv
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS = Path(os.environ.get("DADOS_LIVRO", RAIZ / "dados"))
ZIPS = Path(os.environ.get("COTAHIST_ZIP",
                           RAIZ / "build" / "cotahist"))
SAIDA = DADOS / "bruto" / "b3_cotahist.csv"

URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/"
ANOS = range(2010, 2026)
# AMBV3 é o código da Ambev até 08/11/2013 (depois, ABEV3)
CODIGOS = {"PETR4", "VALE3", "ITUB4", "WEGE3", "ABEV3",
           "AMBV3", "BBAS3", "MGLU3", "BOVA11"}

# Posição de cada campo na linha (início e fim, contados a
# partir de zero, fim excluído) e número de casas decimais.
# No layout da B3 a data ocupa as posições 3 a 10, o código
# 13 a 24, e assim por diante; preços e volume financeiro
# vêm sem vírgula, com duas casas implícitas. O número de
# negócios tem só cinco dígitos e estoura a partir de 100
# mil: fica no extrato como veio, mas o livro não o usa.
CAMPOS = {
    "abertura":   (56, 69, 2),     # PREABE
    "maxima":     (69, 82, 2),     # PREMAX
    "minima":     (82, 95, 2),     # PREMIN
    "fechamento": (108, 121, 2),   # PREULT
    "negocios":   (147, 152, 0),   # TOTNEG
    "quantidade": (152, 170, 0),   # QUATOT
    "volume":     (170, 188, 2),   # VOLTOT, em reais
}


def baixar(ano):
    """Baixa o arquivo do ano, se ainda não existir."""
    destino = ZIPS / f"COTAHIST_A{ano}.ZIP"
    if destino.exists():
        return destino
    ZIPS.mkdir(parents=True, exist_ok=True)
    # o servidor recusa pedidos sem identificação de navegador
    pedido = urllib.request.Request(
        URL + destino.name, headers={"User-Agent": "Mozilla/5.0"})
    parcial = destino.with_suffix(".parcial")
    with urllib.request.urlopen(pedido) as resposta:
        with open(parcial, "wb") as arquivo:
            shutil.copyfileobj(resposta, arquivo)
    parcial.rename(destino)       # só vale depois de completo
    print("baixado:", destino.name)
    return destino


def numero(linha, inicio, fim, casas):
    """Converte um campo numérico de colunas fixas em texto."""
    inteiro = int(linha[inicio:fim])
    if casas == 0:
        return str(inteiro)
    return f"{inteiro / 10 ** casas:.{casas}f}"


def cotacao(linha):
    """Lê uma linha; devolve None se ela não interessa."""
    if linha[0:2] != "01":            # 01 = cotação do dia
        return None
    if linha[24:27] != "010":         # 010 = mercado à vista
        return None
    codigo = linha[12:24].strip()
    if codigo not in CODIGOS:
        return None
    # fator de cotação: 1 quer dizer preço por ação
    assert int(linha[210:217]) == 1, linha
    d = linha[2:10]
    data = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    return [data, codigo] + [numero(linha, *c)
                             for c in CAMPOS.values()]


def extrair(caminho):
    """Cotações dos papéis do livro em um arquivo anual."""
    with zipfile.ZipFile(caminho) as z:
        with z.open(z.namelist()[0]) as texto:
            linhas = (b.decode("latin-1") for b in texto)
            achadas = (cotacao(linha) for linha in linhas)
            return [c for c in achadas if c is not None]


if __name__ == "__main__":
    if SAIDA.exists():
        raise SystemExit(f"{SAIDA} já existe; nada foi feito.")
    tudo = []
    for ano in ANOS:
        tudo += extrair(baixar(ano))
        print(ano, "linhas acumuladas:", len(tudo))
    tudo.sort()                       # por data e por código
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SAIDA, "w", newline="") as f:
        saida = csv.writer(f, lineterminator="\n")
        saida.writerow(["data", "codigo"] + list(CAMPOS))
        saida.writerows(tudo)
    print("gravado:", SAIDA, "com", len(tudo), "linhas")
