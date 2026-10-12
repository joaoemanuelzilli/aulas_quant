"""Estilo único das figuras do livro (cores da Liga, fontes e tamanhos).

Uso, de dentro de livro/scripts/capNN/:

    import sys; sys.path.insert(0, "..")
    from estilo import *
    fig, ax = figura(altura=62)          # largura do bloco de texto (118 mm)
    ...
    salvar(fig, "cap03/minha_figura")    # grava livro/figuras/cap03/minha_figura.pdf
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
FIGURAS = RAIZ / "figuras"

for _f in (RAIZ / "fonts").glob("Inter-*.ttf"):
    font_manager.fontManager.addfont(str(_f))

# cores (a ordem das categóricas é fixa e foi validada para daltonismo)
ROXO, OCRE, VERDE, AZUL, ROSA = "#7B2FBE", "#B97400", "#00A090", "#3F6FD8", "#D1436A"
CATEGORICAS = [ROXO, OCRE, VERDE, AZUL, ROSA]
TINTA, CINZA, CINZA_CLARO, LILAS, ROXO_ESCURO = "#000000", "#6E6E73", "#D7D7DB", "#F3ECFA", "#4A1A78"
SEQ_ROXO = mpl.colors.LinearSegmentedColormap.from_list("seq_roxo", ["#F3ECFA", "#B58BDD", ROXO, "#2B0E47"])
DIV = mpl.colors.LinearSegmentedColormap.from_list("div", [OCRE, "#EFEFF1", ROXO])  # negativo, zero, positivo

MM = 1 / 25.4
L_TEXTO, L_LARGA, L_MARGEM = 118 * MM, 168 * MM, 43 * MM

mpl.rcParams.update({
    "font.family": "Inter", "font.size": 7.6,
    "mathtext.fontset": "stix",
    "axes.titlesize": 8.4, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 7,
    "axes.labelsize": 7.6, "axes.labelcolor": CINZA, "axes.edgecolor": CINZA, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=CATEGORICAS),
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E6E6EA", "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.color": CINZA, "ytick.color": CINZA, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.0, "xtick.major.size": 3, "ytick.major.size": 0,
    "lines.linewidth": 1.25, "lines.solid_capstyle": "round", "lines.markersize": 4,
    "patch.linewidth": 0,
    "legend.frameon": False, "legend.fontsize": 7.2, "legend.handlelength": 1.4,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": None,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42, "axes.unicode_minus": True,
    "axes.formatter.use_locale": False,
})


def figura(altura=62, largura="texto", **kw):
    """Cria figura; `altura` em mm, `largura` em {'texto', 'larga', 'margem'}."""
    w = {"texto": L_TEXTO, "larga": L_LARGA, "margem": L_MARGEM}[largura]
    return plt.subplots(figsize=(w, altura * MM), **kw)


def salvar(fig, nome):
    """Grava figuras/<nome>.pdf (ex.: nome='cap03/histograma')."""
    destino = FIGURAS / f"{nome}.pdf"
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, metadata={"CreationDate": None})
    plt.close(fig)
    print("figura gravada:", destino.relative_to(RAIZ))


def precos(*tickers, inicio=None, fim=None, ajustados=True):
    """Preços de fechamento (ajustados por proventos, por padrão) do conjunto congelado do livro."""
    arq = "precos_ajustados.csv" if ajustados else "precos_sem_ajuste.csv"
    df = pd.read_csv(DADOS / arq, index_col=0, parse_dates=True)
    if tickers:
        df = df[list(tickers)]
    return df.loc[inicio:fim].dropna(how="all")


def rotulo_final(ax, x, y, texto, cor, dx=4, **kw):
    """Rótulo direto no fim de uma linha (substitui a legenda)."""
    ax.annotate(texto, (x, y), xytext=(dx, 0), textcoords="offset points",
                color=TINTA, va="center", fontsize=7.2, **kw)


def pct(ax, eixo="y", casas=0):
    """Formata o eixo como porcentagem (dados em fração: 0.05 -> 5%)."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{casas}f}%".replace(".", ",").replace("-", "\u2212"))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(fmt)


def virgula(ax, eixo="y", casas=1):
    """Formata o eixo com vírgula decimal."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:.{casas}f}".replace(".", ",").replace("-", "\u2212"))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(fmt)
