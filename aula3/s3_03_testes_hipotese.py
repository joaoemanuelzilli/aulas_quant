"""
S3 - Estatística Aplicada ao Mercado Financeiro
Tópico 3: Testes de Hipótese

Caso Prático: Investigação empírica de três questões clássicas de
              finanças quantitativas usando testes de hipótese:
              (A) Retorno médio ≠ 0? (teste t de uma amostra)
              (B) Small caps batem large caps? (teste t de duas amostras)
              (C) Retornos são iid? (Ljung-Box para autocorrelação)

Dependências: pip install yfinance pandas numpy scipy statsmodels matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 0 – FUNDAMENTOS: FRAMEWORK DE TESTE DE HIPÓTESE
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  FRAMEWORK GERAL DE TESTES DE HIPÓTESE")
print("="*65)
print("""
  Passos obrigatórios:
    1. Definir H₀ (hipótese nula) e H₁ (hipótese alternativa)
    2. Escolher estatística de teste e sua distribuição sob H₀
    3. Fixar nível de significância α (risco tipo I: rejeitar H₀ verdadeira)
    4. Calcular p-valor = P(resultado tão ou mais extremo | H₀ verdadeira)
    5. Decisão: rejeitar H₀ se p-valor < α

  Tipos de erro:
    Tipo I  (α): Rejeitar H₀ quando é verdadeira → falso positivo
    Tipo II (β): Não rejeitar H₀ quando é falsa  → falso negativo
    Poder do teste = 1 - β = P(rejeitar H₀ | H₁ verdadeira)

  Conceitos importantes:
    • p-valor NÃO é a probabilidade de H₀ ser verdadeira
    • Significância estatística ≠ significância econômica
    • Múltiplos testes → inflação do erro tipo I (correção de Bonferroni)
""")

# ══════════════════════════════════════════════════════════════════════════════
# DADOS
# ══════════════════════════════════════════════════════════════════════════════
try:
    import yfinance as yf
    tickers_large = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA"]
    tickers_small = ["CSMG3.SA", "TASA4.SA", "FESA4.SA", "BRGE3.SA", "BMKS3.SA"]
    all_tickers = tickers_large + tickers_small
    prices = yf.download(all_tickers, start="2019-01-01", end="2024-12-31",
                         auto_adjust=True, progress=False)["Close"]
    prices.dropna(thresh=int(len(prices)*0.7), axis=1, inplace=True)
    ret_all = np.log(prices / prices.shift(1)).dropna()
    large_tickers = [t for t in tickers_large if t in ret_all.columns]
    small_tickers = [t for t in tickers_small if t in ret_all.columns]
    ret_large = ret_all[large_tickers].mean(axis=1)
    ret_small = ret_all[small_tickers].mean(axis=1) if small_tickers else None
    ibov = yf.download("^BVSP", start="2019-01-01", end="2024-12-31",
               auto_adjust=True, progress=False)["Close"]
    # yfinance pode retornar DataFrame (1 coluna) para ticker unico.
    if isinstance(ibov, pd.DataFrame):
      ibov = ibov.iloc[:, 0]
    ibov.dropna(inplace=True)
    if ibov.empty:
      raise RuntimeError("Download do IBOVESPA retornou serie vazia.")
    ret_ibov = np.log(ibov / ibov.shift(1)).dropna()
    if ret_ibov.empty:
      raise RuntimeError("Serie de retornos do IBOVESPA ficou vazia apos processamento.")
    print(f"Dados baixados: {len(ret_ibov)} pregões")
except Exception:
    raise RuntimeError(
      "Falha ao baixar dados reais via yfinance. "
      "Este script agora usa apenas dados reais."
    )

r = ret_ibov.to_numpy(dtype=float).ravel()

# ══════════════════════════════════════════════════════════════════════════════
# TESTE A – Teste t de uma amostra: μ = 0?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  TESTE A: O retorno médio diário do IBOVESPA é diferente de zero?")
print("="*65)
print("""
  H₀: μ = 0   (mercado não gera retorno médio esperado positivo)
  H₁: μ ≠ 0   (teste bicaudal)

  Estatística:  t = (x̄ - μ₀) / (s / √n)    ~  t(n-1) sob H₀

  Observação: com n grande (~1500), a diferença entre t e Normal é
  mínima pelo CLT, mas usamos t por rigor.
""")

n_obs = len(r)                                  # tamanho da amostra
xbar  = r.mean()                                # média amostral
s     = r.std(ddof=1)                           # desvio-padrão amostral
se    = s / np.sqrt(n_obs)                     # erro padrão da média
t_stat = (xbar - 0) / se                       # estatística t observada
p_val  = 2 * stats.t.sf(abs(t_stat), df=n_obs - 1)   # p-valor bicaudal

# Confirmação com scipy
t_scipy, p_scipy = stats.ttest_1samp(r, popmean=0)   # teste pronto do scipy
t_scipy = float(np.squeeze(t_scipy))                 # t do scipy como número
p_scipy = float(np.squeeze(p_scipy))                 # p-valor do scipy como número

print(f"  n        = {n_obs}")
print(f"  x̄       = {xbar*100:.5f}%  ({xbar*252*100:.2f}% a.a.)")
print(f"  s        = {s*100:.4f}%")
print(f"  SE       = {se*100:.5f}%")
print(f"  t-stat   = {t_stat:.4f}  (scipy: {t_scipy:.4f})")
print(f"  p-valor  = {p_val:.4f}  (scipy: {p_scipy:.4f})")
print(f"  IC 95%   = [{(xbar - 1.96*se)*100:.4f}%, {(xbar + 1.96*se)*100:.4f}%]")

alfa = 0.05
if p_val < alfa:
    print(f"\n  Decisão: REJEITAR H₀ (p={p_val:.4f} < α={alfa})")
    print(f"  → O retorno médio diário é estatisticamente diferente de zero.")
    print(f"  → Porém, {xbar*100:.4f}% por dia pode ser economicamente insignificante.")
else:
    print(f"\n  Decisão: NÃO REJEITAR H₀ (p={p_val:.4f} ≥ α={alfa})")
    print(f"  → Não há evidência suficiente de retorno médio ≠ 0.")

# Poder do teste (sample size para detectar retorno de 0,05% a.a.)
mu_alvo = 0.0005 / 252   # 0,05% a.a. em diário
delta   = abs(mu_alvo - 0) / s
n_necessario = int(np.ceil((stats.norm.ppf(0.975) + stats.norm.ppf(0.80))**2 / delta**2))
print(f"\n  Nota: para detectar μ = {mu_alvo*252*100:.2f}% a.a. com poder 80%,")
print(f"        seriam necessários n ≈ {n_necessario:,} pregões ≈ {n_necessario//252} anos")

# ══════════════════════════════════════════════════════════════════════════════
# TESTE B – Teste t de duas amostras: Small caps batem Large caps?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  TESTE B: Small caps têm retorno médio superior às large caps?")
print("="*65)
print("""
  H₀: μ_small - μ_large = 0   (não há efeito tamanho)
  H₁: μ_small - μ_large > 0   (teste unicaudal — small > large)

  Estatística (Welch — não assume σ igual):
    t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)   ~  t(ν_Welch)

  ν_Welch = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
""")

if ret_small is not None:
    idx_common = ret_large.index.intersection(ret_small.index)
    rl = ret_large.loc[idx_common].values   # retornos das large caps
    rs = ret_small.loc[idx_common].values   # retornos das small caps

    t_b, p_b_two = stats.ttest_ind(rs, rl, equal_var=False)  # teste de Welch
    p_b_one = p_b_two / 2 if t_b > 0 else 1 - p_b_two / 2   # p-valor unicaudal

    print(f"  Small caps — média: {rs.mean()*100:.4f}%/dia  σ: {rs.std()*100:.4f}%/dia")
    print(f"  Large caps — média: {rl.mean()*100:.4f}%/dia  σ: {rl.std()*100:.4f}%/dia")
    print(f"  Diferença  = {(rs.mean()-rl.mean())*100:.4f}%/dia  ({(rs.mean()-rl.mean())*252*100:.2f}% a.a.)")
    print(f"  t-Welch    = {t_b:.4f}")
    print(f"  p-valor (unicaudal) = {p_b_one:.4f}")

    if p_b_one < 0.05:
        print(f"\n  Decisão: REJEITAR H₀ → small caps significativamente > large caps")
        print(f"  → Evidência do 'efeito tamanho' (Fama-French, 1992)")
    else:
        print(f"\n  Decisão: NÃO REJEITAR H₀ → sem evidência de prêmio small cap")
        print(f"  → Efeito pode ser não significativo neste período/amostra")
else:
    print("  (small caps não disponíveis — teste B pulado)")

# ══════════════════════════════════════════════════════════════════════════════
# TESTE C – Ljung-Box: retornos são autocorrelacionados?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  TESTE C: Ljung-Box — retornos têm autocorrelação?")
print("="*65)
print("""
  Motivação: se retornos são autocorrelacionados, o mercado não é
  eficiente na forma fraca → há padrões exploráveis.

  H₀: ρ₁ = ρ₂ = ... = ρₘ = 0   (sem autocorrelação até lag m)
  H₁: ∃ k ≤ m  tal que ρₖ ≠ 0

  Estatística de Ljung-Box:
    Q = n(n+2) · Σₖ₌₁ᵐ  ρ̂ₖ² / (n-k)   ~  χ²(m)  sob H₀

  Onde ρ̂ₖ = autocorrelação amostral no lag k.
""")

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lags_test = [1, 5, 10, 20]                              # lags testados
    lb_ret = acorr_ljungbox(r, lags=lags_test, return_df=True)   # retornos
    lb_ret2 = acorr_ljungbox(r**2, lags=lags_test, return_df=True)  # retornos²

    print("  Ljung-Box — RETORNOS (testa previsibilidade de direção):")
    for lag in lags_test:
        row = lb_ret.loc[lag]
        sig = "***" if row["lb_pvalue"] < 0.01 else ("*" if row["lb_pvalue"] < 0.05 else "")
        print(f"    Lag {lag:2d}: Q={row['lb_stat']:7.2f}  p={row['lb_pvalue']:.4f}  {sig}")

    print("\n  Ljung-Box — RETORNOS² (testa agrupamento de volatilidade / ARCH):")
    for lag in lags_test:
        row = lb_ret2.loc[lag]
        sig = "***" if row["lb_pvalue"] < 0.01 else ("*" if row["lb_pvalue"] < 0.05 else "")
        print(f"    Lag {lag:2d}: Q={row['lb_stat']:7.2f}  p={row['lb_pvalue']:.4f}  {sig}")

    print("""
  Interpretação:
    Retornos:    p alto → sem padrão linear previsível → consistente com HME
    Retornos²:   p baixo → volatilidade autocorrelacionada → efeito ARCH/GARCH!
                 → Motivação para modelos GARCH (ver S4)
  *** p<0,01   * p<0,05
""")
except ImportError:
    print("  statsmodels não disponível. Implementação manual:")
    # ACF manual
    def acf_manual(x, max_lag):
        n = len(x)                         # tamanho da série
        xbar = x.mean()                    # média da série
        c0 = np.sum((x - xbar)**2) / n     # variância base
        return [1.0] + [np.sum((x[:n-k]-xbar)*(x[k:]-xbar))/(n*c0) for k in range(1, max_lag+1)]

    acf_r  = acf_manual(r, 20)     # ACF dos retornos
    acf_r2 = acf_manual(r**2, 20)  # ACF dos retornos²
    n = len(r)                     # tamanho da amostra
    for series_name, acf_vals in [("Retornos", acf_r), ("Retornos²", acf_r2)]:
      print(f"\n  {series_name} — primeiros 5 lags: {[f'{v:.4f}' for v in acf_vals[1:6]]}")
      se_acf = 1/np.sqrt(n)      # banda aproximada de significância
      sig_lags = [k for k,v in enumerate(acf_vals[1:],1) if abs(v) > 2*se_acf]  # lags relevantes
      print(f"  Lags significativos (|ρ| > 2/√n): {sig_lags[:10]}")

# ══════════════════════════════════════════════════════════════════════════════
# TESTE D – Normalidade: Jarque-Bera e Shapiro-Wilk
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  TESTE D: Normalidade dos retornos (JB e Shapiro-Wilk)")
print("="*65)
print("""
  Jarque-Bera:
    JB = n/6 · [S² + (K-3)²/4]   ~  χ²(2)  sob H₀

  onde S = assimetria, K = curtose total.
  H₀: S = 0 e K = 3 (distribuição normal)
""")

jb_stat, jb_p = stats.jarque_bera(r)
print(f"  Jarque-Bera: stat = {jb_stat:.2f}   p-valor = {jb_p:.2e}")
if jb_p < 0.05:
    print("  → REJEITAR normalidade (p < 0,05)")

# Shapiro-Wilk (amostra menor por limitação computacional)
sample_sw = r[:500]
sw_stat, sw_p = stats.shapiro(sample_sw)
print(f"\n  Shapiro-Wilk (n=500): stat = {sw_stat:.5f}   p-valor = {sw_p:.2e}")
if sw_p < 0.05:
    print("  → REJEITAR normalidade")

print("""
  Atenção: com n grande, JB rejeita normalidade mesmo para desvios mínimos.
  Sempre complementar testes formais com análise gráfica (Q-Q Plot, histograma).
""")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
try:
    from statsmodels.graphics.tsaplots import plot_acf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("S3 · Testes de Hipótese — IBOVESPA", fontsize=14,
             fontweight="bold", y=0.99)

# Painel 1: distribuição do t-stat sob H₀ + valor observado
ax = axes[0, 0]
df_plot = n_obs - 1
x_t = np.linspace(-5, 5, 500)
ax.plot(x_t, stats.t.pdf(x_t, df_plot), "b-", lw=2, label=f"t({df_plot})")
ax.fill_between(x_t, stats.t.pdf(x_t, df_plot),
                where=(x_t < -abs(t_stat)) | (x_t > abs(t_stat)),
                color="red", alpha=0.3, label=f"p-valor={p_val:.4f}")
ax.axvline(t_stat, color="red", lw=2, ls="--", label=f"t-obs={t_stat:.2f}")
ax.axvline(-t_stat, color="red", lw=2, ls="--")
ax.set_title("Teste A: distribuição t sob H₀", fontsize=10)
ax.set_xlabel("t-estatística")
ax.legend(fontsize=8)

# Painel 2: bootstrap da média (validação não-paramétrica)
ax = axes[0, 1]
n_boot = 5000
boot_means = np.array([np.random.choice(r, size=n_obs, replace=True).mean()
                       for _ in range(n_boot)])
ax.hist(boot_means, bins=60, density=True, color="#7B2D8B", alpha=0.6)
ax.axvline(0, color="red", lw=2, ls="--", label="H₀: μ=0")
ax.axvline(xbar, color="green", lw=2, label=f"x̄={xbar*100:.4f}%")
pct_above_zero = (boot_means > 0).mean()
ax.set_title(f"Bootstrap da média (P(μ̂>0)={pct_above_zero:.1%})", fontsize=10)
ax.set_xlabel("Médias bootstrap")
ax.legend(fontsize=8)

# Painel 3: ACF retornos
ax = axes[0, 2]
if HAVE_SM:
    plot_acf(r, lags=30, ax=ax, color="#2196F3", alpha=0.5)
else:
    acf_vals = acf_manual(r, 30)
    ax.bar(range(31), acf_vals, color="#2196F3", alpha=0.7)
    ax.axhline(2/np.sqrt(n_obs), color="red", ls="--")
    ax.axhline(-2/np.sqrt(n_obs), color="red", ls="--")
ax.set_title("ACF — Retornos diários", fontsize=10)

# Painel 4: ACF retornos²
ax = axes[1, 0]
if HAVE_SM:
    plot_acf(r**2, lags=30, ax=ax, color="#E53935", alpha=0.5)
else:
    acf_vals2 = acf_manual(r**2, 30)
    ax.bar(range(31), acf_vals2, color="#E53935", alpha=0.7)
    ax.axhline(2/np.sqrt(n_obs), color="black", ls="--")
    ax.axhline(-2/np.sqrt(n_obs), color="black", ls="--")
ax.set_title("ACF — Retornos² (volatilidade)", fontsize=10)

# Painel 5: p-valores Ljung-Box ao longo dos lags
ax = axes[1, 1]
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lags_all = list(range(1, 31))
    lb_all = acorr_ljungbox(r, lags=lags_all, return_df=True)
    lb2_all = acorr_ljungbox(r**2, lags=lags_all, return_df=True)
    ax.semilogy(lags_all, lb_all["lb_pvalue"], "b-o", ms=4, label="Retornos")
    ax.semilogy(lags_all, lb2_all["lb_pvalue"], "r-s", ms=4, label="Retornos²")
except Exception:
    ax.text(0.3, 0.5, "statsmodels\nnecessário", transform=ax.transAxes, fontsize=10)
ax.axhline(0.05, color="black", ls="--", lw=1, label="α=5%")
ax.set_title("Ljung-Box p-valores por lag", fontsize=10)
ax.set_xlabel("Lag")
ax.set_ylabel("p-valor (escala log)")
ax.legend(fontsize=8)

# Painel 6: erro tipo I vs tamanho de amostra
ax = axes[1, 2]
ns = np.arange(50, 2001, 50)
# Probabilidade de rejeitar H₀ verdadeira (deve ser α=5% — é constante)
# Mas o poder aumenta com n: simular rejeições quando μ=0.02%/dia (pequeno efeito)
mu_alt = 0.0002
power_curve = []
for n_sim in ns:
    rejeicoes = 0
    for _ in range(500):
        sample = np.random.normal(mu_alt, s, n_sim)
        _, p = stats.ttest_1samp(sample, 0)
        if p < 0.05:
            rejeicoes += 1
    power_curve.append(rejeicoes / 500)
ax.plot(ns, power_curve, color="#7B2D8B", lw=2)
ax.axhline(0.80, color="red", ls="--", label="Poder 80%")
ax.set_title(f"Curva de poder (μ_alt={mu_alt*100:.2f}%/dia, α=5%)", fontsize=10)
ax.set_xlabel("Tamanho da amostra n")
ax.set_ylabel("Poder do teste")
ax.legend(fontsize=8)

plt.tight_layout()
output_file = OUTPUT_DIR / "s3_03_testes_hipotese.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
plt.show()
print(f"[✓] Gráfico salvo: {output_file}")
