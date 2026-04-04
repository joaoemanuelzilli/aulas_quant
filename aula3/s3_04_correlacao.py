"""
S3 - Estatística Aplicada ao Mercado Financeiro
Tópico 4: Correlação

Caso Prático: Análise completa de correlação entre ativos do IBOVESPA
              para construção de portfólios diversificados, incluindo:
              (A) Pearson, Spearman, Kendall
              (B) Correlação rolante (instabilidade temporal)
              (C) Correlação em crise vs. normalidade
              (D) Correlação vs. causalidade (Granger)

Dependências: pip install yfinance pandas numpy scipy statsmodels matplotlib seaborn
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

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 0 – FUNDAMENTOS MATEMÁTICOS
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  FUNDAMENTOS: MEDIDAS DE CORRELAÇÃO")
print("="*65)
print("""
  PEARSON (correlação linear):
    ρ(X,Y) = Cov(X,Y) / [σ(X)·σ(Y)]
           = Σ(xᵢ-x̄)(yᵢ-ȳ) / √[Σ(xᵢ-x̄)² · Σ(yᵢ-ȳ)²]

    • -1 ≤ ρ ≤ 1
    • Mede dependência LINEAR
    • Sensível a outliers
    • Premissa: distribuição conjunta normal (mas é robusto)

  SPEARMAN (correlação de ranks):
    ρₛ = Pearson(rank(X), rank(Y))
       = 1 - 6Σdᵢ² / [n(n²-1)]   onde dᵢ = rank(xᵢ) - rank(yᵢ)

    • Mede dependência MONOTÔNICA (não necessariamente linear)
    • Robusto a outliers e distribuições assimétricas
    • Preferível quando retornos têm caudas pesadas

  KENDALL (concordância):
    τ = (# pares concordantes - # pares discordantes) / C(n,2)

    • Mais interpretável: τ = P(concordante) - P(discordante)
    • Robusto e preferível para amostras pequenas

  TESTE DE SIGNIFICÂNCIA (H₀: ρ = 0):
    t = r√(n-2) / √(1-r²)   ~  t(n-2)   (Pearson)
""")

# ══════════════════════════════════════════════════════════════════════════════
# DADOS
# ══════════════════════════════════════════════════════════════════════════════
TICKERS = {
    "PETR4": "PETR4.SA",  # Petróleo
    "VALE3": "VALE3.SA",  # Mineração
    "ITUB4": "ITUB4.SA",  # Banco
    "WEGE3": "WEGE3.SA",  # Indústria
    "MGLU3": "MGLU3.SA",  # Varejo
    "BBAS3": "BBAS3.SA",  # Banco estatal
}

try:
    import yfinance as yf
    raw = yf.download(list(TICKERS.values()), start="2018-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)["Close"]
    raw.columns = [k for k, v in TICKERS.items() if v in raw.columns]
    raw.dropna(thresh=int(len(raw)*0.8), axis=1, inplace=True)
    raw.fillna(method="ffill", inplace=True)
    raw.dropna(inplace=True)
    print(f"Dados: {len(raw)} pregões | Ativos: {list(raw.columns)}")
except Exception:
    print("Usando dados sintéticos")
    n = 1700
    dates = pd.bdate_range("2018-01-01", periods=n)
  # Simular retornos com estrutura de correlação realista.
    cov_true = np.array([
        [1.00, 0.55, 0.40, 0.30, 0.25, 0.42],
        [0.55, 1.00, 0.38, 0.28, 0.20, 0.35],
        [0.40, 0.38, 1.00, 0.35, 0.30, 0.75],
        [0.30, 0.28, 0.35, 1.00, 0.22, 0.30],
        [0.25, 0.20, 0.30, 0.22, 1.00, 0.28],
        [0.42, 0.35, 0.75, 0.30, 0.28, 1.00],
    ])
    vols = np.array([0.025, 0.022, 0.018, 0.016, 0.030, 0.017])
    cov_matrix = np.outer(vols, vols) * cov_true
    L = np.linalg.cholesky(cov_matrix)
    ret_arr = (L @ np.random.randn(6, n)).T + 0.0003
    ret_sim = pd.DataFrame(ret_arr, index=dates, columns=list(TICKERS.keys()))
    # Converte retornos simulados em serie de precos positiva para manter
    # a mesma pipeline didatica usada com dados reais.
    raw = 100 * np.exp(ret_sim.cumsum())

ret = np.log(raw / raw.shift(1)).dropna()
assets = list(ret.columns)

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 1 – MATRIZ DE CORRELAÇÃO (3 métodos)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  PARTE 1: Matrizes de Correlação — Pearson vs Spearman vs Kendall")
print("="*65)

corr_pearson  = ret.corr(method="pearson")
corr_spearman = ret.corr(method="spearman")
corr_kendall  = ret.corr(method="kendall")

print("\n  Pearson:")
print(corr_pearson.round(3).to_string())
print("\n  Spearman:")
print(corr_spearman.round(3).to_string())
print("\n  Kendall:")
print(corr_kendall.round(3).to_string())

# Par de maior discrepância Pearson vs Spearman
diff = (corr_pearson - corr_spearman).abs()
diff_values = diff.to_numpy(copy=True)
np.fill_diagonal(diff_values, 0)
max_pair = np.unravel_index(diff_values.argmax(), diff_values.shape)
print(f"\n  Maior discrepância Pearson vs Spearman:")
print(f"  Par: {assets[max_pair[0]]} — {assets[max_pair[1]]}")
print(f"  Pearson: {corr_pearson.iloc[max_pair]:.3f}  |  Spearman: {corr_spearman.iloc[max_pair]:.3f}")
print(f"  → Sugere relação não-linear ou influência de outliers nesse par")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 2 – SIGNIFICÂNCIA ESTATÍSTICA DAS CORRELAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  PARTE 2: Significância das correlações de Pearson")
print("="*65)
print("""
  Teste: H₀: ρ = 0  vs  H₁: ρ ≠ 0
  t = r√(n-2)/√(1-r²)  ~  t(n-2)
""")
n_obs = len(ret)
print(f"  {'Par':<20} {'ρ_Pearson':>10}  {'t-stat':>8}  {'p-valor':>10}  {'Sig?':>6}")
print(f"  {'-'*60}")
for i, a1 in enumerate(assets):
    for j, a2 in enumerate(assets):
        if j <= i:
            continue
        rho = corr_pearson.loc[a1, a2]
        t_val = rho * np.sqrt(n_obs - 2) / np.sqrt(1 - rho**2)
        p_val = 2 * stats.t.sf(abs(t_val), df=n_obs - 2)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        print(f"  {a1}–{a2:<14} {rho:>10.3f}  {t_val:>8.2f}  {p_val:>10.4f}  {sig:>6}")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 3 – CORRELAÇÃO ROLANTE (instabilidade temporal)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  PARTE 3: Correlação rolante — instabilidade temporal")
print("="*65)
print("""
  Correlação não é estável ao longo do tempo!
  Em crises, correlações tendem a subir (efeito "tudo cai junto"),
  reduzindo os benefícios da diversificação exatamente quando mais importa.

  ρ_t(janela) = Pearson calculado na janela [t-janela, t]
""")

WINDOW = 63   # ≈ 3 meses úteis

# Pares mais representativos
pairs = [("PETR4", "VALE3"), ("ITUB4", "BBAS3"), ("PETR4", "ITUB4")]
pairs = [(a, b) for a, b in pairs if a in assets and b in assets]
if not pairs and len(assets) >= 2:
    pairs = [(assets[0], assets[1])]

roll_corrs = {}
for a1, a2 in pairs:
    rc = ret[[a1, a2]].rolling(WINDOW).corr().unstack()[(a1, a2)].dropna()
    if rc.empty:
        print(f"  {a1}–{a2}: série insuficiente para janela de {WINDOW} dias")
        continue
    roll_corrs[f"{a1}–{a2}"] = rc
    print(f"  {a1}–{a2}: ρ médio={rc.mean():.3f}  min={rc.min():.3f}  max={rc.max():.3f}"
          f"  std={rc.std():.3f}")
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  PARTE 4: Correlação em regimes — crise vs. normalidade")
print("="*65)

# Regra principal: dias em que o retorno medio do portfolio cai abaixo de μ-2σ.
# Se essa regra gerar poucos pontos, usa corte por percentil para manter
# comparacao de regimes estatisticamente informativa.
mean_ret_port = ret.mean(axis=1)
threshold = mean_ret_port.mean() - 2 * mean_ret_port.std()
crisis_mask = mean_ret_port < threshold
crisis_rule = "μ-2σ"
min_crisis_obs = max(30, int(0.05 * len(ret)))
if crisis_mask.sum() < min_crisis_obs:
    q = 0.15
    threshold = mean_ret_port.quantile(q)
    crisis_mask = mean_ret_port <= threshold
    crisis_rule = f"percentil {int(q*100)}"

normal_mask = ~crisis_mask

print(f"  Dias classificados como 'crise' ({crisis_rule}): "
  f"{crisis_mask.sum()} ({crisis_mask.mean()*100:.1f}%)")

corr_crisis = ret[crisis_mask].corr(method="pearson") if crisis_mask.sum() > 20 else None
corr_normal = ret[normal_mask].corr(method="pearson")

if corr_crisis is not None:
    print("\n  Correlação média entre pares — NORMALIDADE vs. CRISE:")
    for i, a1 in enumerate(assets):
        for j, a2 in enumerate(assets):
            if j <= i:
                continue
            rho_n = corr_normal.loc[a1, a2]
            rho_c = corr_crisis.loc[a1, a2]
            delta = rho_c - rho_n
            arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
            print(f"  {a1}–{a2:<12}: Normal={rho_n:.3f}  Crise={rho_c:.3f}  "
                  f"Δ={delta:+.3f} {arrow}")
    print("""
  Fenômeno bem documentado: correlações sobem em crises.
  → Diversificação funciona bem no dia a dia, mas falha quando mais necessária.
  → Solução: incluir ativos com correlação negativa em crises
    (ex.: ouro, dólar, puts, bonds de países desenvolvidos).
""")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 5 – CORRELAÇÃO VS. CAUSALIDADE (Granger)
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  PARTE 5: Correlação ≠ Causalidade — Granger Causality")
print("="*65)
print("""
  Granger Causality:
    X Granger-causa Y se X ajuda a prever Y além do que Y prevê a si mesmo.

  Modelo restrito:   Yₜ = α + Σᵢ βᵢYₜ₋ᵢ + εₜ
  Modelo irrestrito: Yₜ = α + Σᵢ βᵢYₜ₋ᵢ + Σⱼ γⱼXₜ₋ⱼ + εₜ

  H₀: γ₁ = γ₂ = ... = γₘ = 0   (X não Granger-causa Y)
  Estatística F = [(RSS_R - RSS_U)/m] / [RSS_U/(n-2m-1)]   ~  F(m, n-2m-1)

  Nota: Granger causality é previsibilidade, não causalidade real.
""")

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    pair_gc = (assets[0], assets[1]) if len(assets) >= 2 else None
    if pair_gc:
        a1, a2 = pair_gc
        data_gc = ret[[a1, a2]].dropna()
        print(f"\n  Teste: {a1} Granger-causa {a2}?")
        gc_results = grangercausalitytests(data_gc, maxlag=5, verbose=False)
        print(f"  {'Lag':>5}  {'F-stat':>10}  {'p-valor':>10}  {'Sig?':>6}")
        for lag, res in gc_results.items():
            f_stat = res[0]['ssr_ftest'][0]
            p_gc   = res[0]['ssr_ftest'][1]
            sig = "***" if p_gc < 0.001 else ("**" if p_gc < 0.01 else
                  ("*" if p_gc < 0.05 else ""))
            print(f"  {lag:>5}  {f_stat:>10.3f}  {p_gc:>10.4f}  {sig:>6}")
except ImportError:
    print("  (statsmodels não disponível para Granger causality)")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 6 – IMPLICAÇÃO PARA PORTFÓLIOS (diversificação)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  PARTE 6: Impacto da correlação na volatilidade do portfólio")
print("="*65)
print("""
  Para portfólio equally-weighted de n ativos com σᵢ ≈ σ e ρᵢⱼ ≈ ρ:

    σ_port = σ · √[(1/n) + ρ · (1 - 1/n)]

  Limites:
    ρ = 1  →  σ_port = σ        (sem diversificação)
    ρ = 0  →  σ_port = σ/√n     (diversificação máxima sem correlação)
    ρ = -1 →  σ_port → 0        (hedge perfeito, impossível na prática)
""")

sigma_avg = ret.std().mean()
rho_avg   = corr_pearson.values[np.triu_indices_from(corr_pearson.values, k=1)].mean()
print(f"  σ médio diário dos ativos: {sigma_avg*100:.3f}%")
print(f"  ρ médio entre pares:       {rho_avg:.3f}")
print()
print(f"  {'n ativos':>10}  {'σ_port (ρ_real)':>16}  {'σ_port (ρ=0)':>14}  {'Redução (ρ_real)':>18}")
for n_a in [1, 2, 3, 5, 10, 20]:
    sig_real = sigma_avg * np.sqrt(1/n_a + rho_avg*(1 - 1/n_a))
    sig_zero = sigma_avg / np.sqrt(n_a)
    red = (1 - sig_real/sigma_avg)*100
    print(f"  {n_a:>10}  {sig_real*100:>14.4f}%  {sig_zero*100:>12.4f}%  {red:>16.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 16))
fig.suptitle("S3 · Correlação — Análise Completa de Portfólio",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Painel 1: Heatmap Pearson
ax1 = fig.add_subplot(gs[0, 0])
mask = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, ax=ax1, square=True,
            cbar_kws={"shrink": 0.8})
ax1.set_title("Correlação de Pearson", fontsize=10)

# Painel 2: Heatmap Spearman
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, ax=ax2, square=True,
            cbar_kws={"shrink": 0.8})
ax2.set_title("Correlação de Spearman", fontsize=10)

# Painel 3: Pearson vs Spearman (scatter)
ax3 = fig.add_subplot(gs[0, 2])
p_vals = corr_pearson.values[np.triu_indices_from(corr_pearson.values, k=1)]
s_vals = corr_spearman.values[np.triu_indices_from(corr_spearman.values, k=1)]
ax3.scatter(p_vals, s_vals, s=60, color="#7B2D8B", alpha=0.8, edgecolors="white")
ax3.plot([-1,1], [-1,1], "k--", lw=1, label="ρ_P = ρ_S")
ax3.set_xlabel("Pearson")
ax3.set_ylabel("Spearman")
ax3.set_title("Pearson vs Spearman (todos os pares)", fontsize=10)
ax3.legend(fontsize=8)

# Painel 4: Correlações rolantes
ax4 = fig.add_subplot(gs[1, :])
COLORS_ROLL = ["#7B2D8B", "#E53935", "#2196F3", "#FF6F00"]
for (pair_name, rc), color in zip(roll_corrs.items(), COLORS_ROLL):
    ax4.plot(rc.index, rc.values, lw=1.2, color=color,
             label=pair_name, alpha=0.85)
ax4.axhline(0, color="black", lw=0.8, ls="--")
ax4.fill_between(ret.index,
                 -2/np.sqrt(WINDOW), 2/np.sqrt(WINDOW),
                 alpha=0.1, color="gray", label="IC 95% H₀:ρ=0")
ax4.set_title(f"Correlação Rolante de Pearson (janela={WINDOW} dias úteis ≈ 3 meses)",
              fontsize=11)
ax4.set_xlabel("Data")
ax4.set_ylabel("ρ")
if roll_corrs:
  ax4.legend(fontsize=8, ncol=4)
else:
  ax4.text(0.33, 0.52, "Sem dados suficientes\npara correlação rolante",
       transform=ax4.transAxes)

# Painel 5: Normalidade vs crise
ax5 = fig.add_subplot(gs[2, 0])
if corr_crisis is not None:
    p_norm_all = corr_normal.values[np.triu_indices_from(corr_normal.values, k=1)]
    p_cris_all = corr_crisis.values[np.triu_indices_from(corr_crisis.values, k=1)]
    ax5.scatter(p_norm_all, p_cris_all, s=60, color="#E53935", alpha=0.8)
    lim = max(abs(p_norm_all).max(), abs(p_cris_all).max()) + 0.05
    ax5.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="igualdade")
    ax5.set_xlabel("ρ em normalidade")
    ax5.set_ylabel("ρ em crise")
    ax5.set_title("Crise vs. Normalidade", fontsize=10)
    ax5.legend(fontsize=8)
else:
    ax5.text(0.3, 0.5, "Dados insuficientes\npara regime de crise",
             transform=ax5.transAxes)

# Painel 6: Scatter maior par
ax6 = fig.add_subplot(gs[2, 1])
if len(assets) >= 2:
    a1, a2 = assets[0], assets[1]
    ax6.scatter(ret[a1], ret[a2], s=4, alpha=0.3, color="#5C6BC0")
    slope, intercept, r_val, _, _ = stats.linregress(ret[a1], ret[a2])
    xfit = np.linspace(ret[a1].min(), ret[a1].max(), 100)
    ax6.plot(xfit, slope*xfit + intercept, "r-", lw=2,
             label=f"ρ={r_val:.3f}")
    ax6.set_xlabel(a1)
    ax6.set_ylabel(a2)
    ax6.set_title(f"Scatter: {a1} vs {a2}", fontsize=10)
    ax6.legend(fontsize=9)

# Painel 7: Volatilidade portfólio vs. n ativos (diversificação)
ax7 = fig.add_subplot(gs[2, 2])
n_range = np.arange(1, 31)
sig_div_real = sigma_avg * np.sqrt(1/n_range + rho_avg*(1 - 1/n_range))
sig_div_zero = sigma_avg / np.sqrt(n_range)
ax7.plot(n_range, sig_div_real*100, "r-", lw=2, label=f"ρ={rho_avg:.2f} (real)")
ax7.plot(n_range, sig_div_zero*100, "g--", lw=2, label="ρ=0 (ideal)")
ax7.axhline(sigma_avg*np.sqrt(rho_avg)*100, color="gray", ls=":",
            label=f"Limite ρ→∞: {sigma_avg*np.sqrt(rho_avg)*100:.3f}%")
ax7.set_xlabel("Número de ativos no portfólio")
ax7.set_ylabel("σ_port diária (%)")
ax7.set_title("Benefício da diversificação", fontsize=10)
ax7.legend(fontsize=8)

output_file = OUTPUT_DIR / "s3_04_correlacao.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
plt.show()
print(f"[✓] Gráfico salvo: {output_file}")

print("""
╔══════════════════════════════════════════════════════════════════╗
║  RESUMO — CORRELAÇÃO NO MERCADO FINANCEIRO                       ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Use Spearman quando retornos têm caudas pesadas (quase       ║
║     sempre em ações brasileiras).                                ║
║  2. Correlação é instável: calcule sempre a janela rolante       ║
║     antes de tomar decisões de portfólio.                        ║
║  3. Correlações sobem em crises → diversificação falha           ║
║     exatamente quando mais importa.                              ║
║  4. Pearson captura apenas dependência linear; Copulas           ║
║     capturam dependência de cauda (ver cursos avançados).        ║
║  5. Correlação ≠ causalidade. Use Granger com cautela.           ║
╚══════════════════════════════════════════════════════════════════╝
""")
