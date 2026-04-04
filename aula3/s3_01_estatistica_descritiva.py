"""
S3 - Estatística Aplicada ao Mercado Financeiro
Tópico 1: Estatística Descritiva

Caso Prático: Análise descritiva completa dos retornos diários de
              PETR4, VALE3 e ITUB4 (2020–2024), identificando
              características de risco e retorno de cada ativo.

Dependências: pip install yfinance pandas numpy scipy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Reprodutibilidade ──────────────────────────────────────────────────────────
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. DOWNLOAD DOS DADOS ──────────────────────────────────────────────────────
try:
    import yfinance as yf
    tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
    raw = yf.download(tickers, start="2020-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)["Close"]
    raw.dropna(inplace=True)
    print(f"Dados baixados: {len(raw)} pregões | {raw.index[0].date()} → {raw.index[-1].date()}")
    USE_REAL = True
except Exception as exc:
    raise RuntimeError(
        "Falha ao baixar dados reais via yfinance. Verifique conexao e ticker(s)."
    ) from exc

# Retornos log-diários (já vistos na S1)
ret = np.log(raw / raw.shift(1)).dropna()
ret.columns = ["PETR4", "VALE3", "ITUB4"]

# ── 2. MEDIDAS DE TENDÊNCIA CENTRAL ───────────────────────────────────────────
print("\n" + "="*65)
print("  MEDIDAS DE TENDÊNCIA CENTRAL  (retornos diários, %)")
print("="*65)

for col in ret.columns:
    r = ret[col]
    media    = r.mean()
    mediana  = r.median()
    # Moda aproximada via kernel (série contínua → moda = pico da densidade)
    kde_x    = np.linspace(r.min(), r.max(), 1000)
    kde_y    = stats.gaussian_kde(r)(kde_x)
    moda_kde = kde_x[np.argmax(kde_y)]
    print(f"\n{col}:")
    print(f"  Média   = {media*100:+.4f}%   ({media*252*100:+.2f}% a.a. anualizada)")
    print(f"  Mediana = {mediana*100:+.4f}%")
    print(f"  Moda≈   = {moda_kde*100:+.4f}%  (via KDE)")
    print(f"  Interpretação: média {'>' if media > mediana else '<'} mediana → "
          f"assimetria {'positiva (cauda direita)' if media > mediana else 'negativa (cauda esquerda)'}")

# ── 3. MEDIDAS DE DISPERSÃO ────────────────────────────────────────────────────
print("\n" + "="*65)
print("  MEDIDAS DE DISPERSÃO")
print("="*65)

for col in ret.columns:
    r = ret[col]
    std_d   = r.std()
    std_a   = std_d * np.sqrt(252)     # Volatilidade anualizada (raiz do tempo)
    var_d   = r.var()
    amp     = r.max() - r.min()
    iqr     = r.quantile(0.75) - r.quantile(0.25)
    cv      = std_d / abs(r.mean())    # Coef. de variação (risco por unidade de retorno)
    print(f"\n{col}:")
    print(f"  Desvio-padrão diário  = {std_d*100:.3f}%")
    print(f"  Volatilidade anual    = {std_a*100:.2f}%  (x√252)")
    print(f"  Variância diária      = {var_d*1e4:.4f} (x10⁻⁴)")
    print(f"  Amplitude (max-min)   = {amp*100:.2f}%")
    print(f"  IQR (Q3-Q1)          = {iqr*100:.3f}%")
    print(f"  Coef. de variação     = {cv:.1f}x  (risco/retorno bruto)")

# ── 4. FORMA DA DISTRIBUIÇÃO ───────────────────────────────────────────────────
print("\n" + "="*65)
print("  FORMA DA DISTRIBUIÇÃO  (assimetria e curtose)")
print("="*65)

resumo = []
for col in ret.columns:
    r = ret[col]
    skew = r.skew()           # Assimetria de Fisher (g1)
    kurt = r.kurtosis()       # Curtose em excesso (g2); Normal → 0
    kurt_total = kurt + 3     # Curtose total; Normal → 3
    jb_stat, jb_p = stats.jarque_bera(r)
    resumo.append({
        "Ativo"      : col,
        "Assimetria" : round(skew, 3),
        "Curtose exc": round(kurt, 3),
        "Curtose tot": round(kurt_total, 3),
        "JB stat"    : round(jb_stat, 1),
        "p-valor JB" : f"{jb_p:.2e}",
        "Normal?"    : "Não" if jb_p < 0.05 else "Sim",
    })

df_resumo = pd.DataFrame(resumo).set_index("Ativo")
print(df_resumo.to_string())
print("""
Interpretação:
  Assimetria < 0  → cauda esquerda pesada (perdas extremas mais frequentes)
  Curtose exc > 0 → leptocúrtica (mais picos e caudas do que a Normal)
  JB p < 0,05     → rejeita normalidade → distribuições com caudas pesadas!
""")

# ── 5. PERCENTIS, QUARTIS E OUTLIERS ──────────────────────────────────────────
print("="*65)
print("  PERCENTIS & DETECÇÃO DE OUTLIERS")
print("="*65)

for col in ret.columns:
    r = ret[col]
    q1, q3 = r.quantile(0.25), r.quantile(0.75)
    iqr    = q3 - q1
    lower  = q1 - 1.5 * iqr
    upper  = q3 + 1.5 * iqr
    outliers = r[(r < lower) | (r > upper)]
    p1, p5, p95, p99 = r.quantile([0.01, 0.05, 0.95, 0.99])
    print(f"\n{col}:")
    print(f"  P1={p1*100:.2f}%  P5={p5*100:.2f}%  P95={p95*100:.2f}%  P99={p99*100:.2f}%")
    print(f"  Outliers IQR (|r| > fence): {len(outliers)} dias ({len(outliers)/len(r)*100:.1f}%)")
    print(f"  Maior queda: {r.min()*100:.2f}%  em {r.idxmin().date()}")
    print(f"  Maior alta:  {r.max()*100:.2f}%  em {r.idxmax().date()}")

# ── 6. VISUALIZAÇÕES ──────────────────────────────────────────────────────────
COLORS = {"PETR4": "#7B2D8B", "VALE3": "#2196F3", "ITUB4": "#FF6F00"}

fig = plt.figure(figsize=(18, 14))
fig.suptitle("S3 · Estatística Descritiva — Retornos Diários (2020–2024)",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

for i, col in enumerate(ret.columns):
    r = ret[col]
    color = COLORS[col]

    # --- Histograma com KDE e Normal teórica ---
    ax = fig.add_subplot(gs[i, 0])
    ax.hist(r, bins=80, density=True, color=color, alpha=0.45,
            label="Retornos obs.")
    xg = np.linspace(r.min(), r.max(), 400)
    ax.plot(xg, stats.norm.pdf(xg, r.mean(), r.std()),
            "k--", lw=1.5, label="Normal teórica")
    ax.plot(xg, stats.gaussian_kde(r)(xg),
            color=color, lw=2, label="KDE empírica")
    ax.set_title(f"{col} — Histograma + KDE", fontsize=10)
    ax.set_xlabel("Retorno diário")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=7)

    # --- Boxplot com anotações ---
    ax2 = fig.add_subplot(gs[i, 1])
    bp = ax2.boxplot(r, vert=True, patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.5),
                     medianprops=dict(color="black", lw=2),
                     flierprops=dict(marker=".", markersize=2, alpha=0.4))
    q1, med, q3 = r.quantile(0.25), r.median(), r.quantile(0.75)
    ax2.axhline(r.mean(), color="red", ls="--", lw=1, label=f"Média={r.mean()*100:.3f}%")
    ax2.set_title(f"{col} — Boxplot", fontsize=10)
    ax2.set_ylabel("Retorno diário")
    ax2.legend(fontsize=7)
    ax2.set_xticks([])

    # --- Q-Q Plot ---
    ax3 = fig.add_subplot(gs[i, 2])
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    ax3.scatter(osm, osr, s=6, color=color, alpha=0.5, label="Dados")
    ax3.plot(osm, slope * np.array(osm) + intercept,
             "k--", lw=1.5, label="Normal ref.")
    ax3.set_title(f"{col} — Q-Q Plot", fontsize=10)
    ax3.set_xlabel("Quantis teóricos (Normal)")
    ax3.set_ylabel("Quantis observados")
    ax3.legend(fontsize=7)

output_file = OUTPUT_DIR / "s3_01_estatistica_descritiva.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[✓] Gráfico salvo: {output_file}")

# ── 7. TABELA RESUMO FINAL ────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RESUMO EXECUTIVO")
print("="*65)
sumtab = pd.DataFrame({
    col: {
        "Média diária (%)":    f"{ret[col].mean()*100:+.4f}",
        "Retorno anual (%)":   f"{ret[col].mean()*252*100:+.2f}",
        "Vol. anual (%)":      f"{ret[col].std()*np.sqrt(252)*100:.2f}",
        "Sharpe aprox. (rf=0)": f"{(ret[col].mean()*252)/(ret[col].std()*np.sqrt(252)):.3f}",
        "Assimetria":          f"{ret[col].skew():.3f}",
        "Curtose exc.":        f"{ret[col].kurtosis():.3f}",
        "Max drawdown (1 dia)": f"{ret[col].min()*100:.2f}%",
    }
    for col in ret.columns
}).T
print(sumtab.to_string())
print("""
Conclusão do Caso Prático:
  • Todos os ativos exibem curtose positiva → caudas mais pesadas que a Normal.
  • Assimetria negativa → perdas extremas ocorrem com mais frequência/magnitude
    do que ganhos extremos → não é seguro usar σ como única medida de risco.
  • Q-Q plots confirmam: os extremos (outliers) se afastam da linha Normal,
    especialmente no lado esquerdo (crashes de mercado).
  • Implicação prática: modelos que assumem normalidade (ex.: VaR paramétrico
    simples) subestimarão o risco real → ver S5.
""")
