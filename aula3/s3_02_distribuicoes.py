"""
S3 - Estatística Aplicada ao Mercado Financeiro
Tópico 2: Distribuições — Normal, t-Student e Caudas Pesadas

Caso Prático: Ajuste de distribuições aos retornos do IBOVESPA e
              comparação do VaR estimado por cada distribuição,
              demonstrando por que a Normal subestima o risco de cauda.

Dependências: pip install yfinance pandas numpy scipy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.special import gamma as Γ
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 0 – DADOS
# ══════════════════════════════════════════════════════════════════════════════
try:
  import yfinance as yf

  ibov = yf.download("^BVSP", start="2010-01-01", end="2024-12-31",
             auto_adjust=True, progress=False)["Close"]
  # yfinance pode retornar DataFrame (1 coluna) para ticker unico.
  if isinstance(ibov, pd.DataFrame):
    ibov = ibov.iloc[:, 0]

  ibov.dropna(inplace=True)
  if ibov.empty:
    raise RuntimeError("Download do IBOVESPA retornou serie vazia.")
  ret = np.log(ibov / ibov.shift(1)).dropna()
  print(f"IBOVESPA: {len(ret)} retornos diários  "
      f"({ret.index[0].date()} → {ret.index[-1].date()})")
except Exception:
  raise RuntimeError(
    "Falha ao baixar dados reais do IBOVESPA via yfinance. "
    "Este script agora usa apenas dados reais."
  )

r = ret.to_numpy(dtype=float).ravel()  # vetor 1D para calculos e scipy

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 1 – REVISÃO MATEMÁTICA DAS DISTRIBUIÇÕES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  DISTRIBUIÇÃO NORMAL  N(μ, σ²)")
print("="*65)
print("""
  PDF:   f(x) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²))

  Propriedades-chave para finanças:
    • Simétrica → P(r < -x) = P(r > +x)   [não vale em mercados!]
    • Soma de normais independentes é normal → base do CLT
    • Curtose excesso = 0  (distribuição mesocúrtica)
    • 68-95-99,7% dos valores dentro de ±1σ, ±2σ, ±3σ
    • Limitação: subestima eventos extremos (crashes/rallies)
""")

# Ajuste Normal
mu_n, sig_n = stats.norm.fit(r)
print(f"  Ajuste ao IBOV:  μ = {mu_n*100:.4f}%   σ = {sig_n*100:.4f}%")

print("\n" + "="*65)
print("  DISTRIBUIÇÃO t-STUDENT  t(ν)")
print("="*65)
print("""
  PDF:   f(x|ν) = Γ((ν+1)/2) / [√(νπ)·Γ(ν/2)] · (1 + x²/ν)^(-(ν+1)/2)

  Para localização e escala (t de três parâmetros):
         f(x|μ,σ,ν) = f((x-μ)/σ | ν) / σ

  Propriedades-chave:
    • ν → ∞  ⟹  t(ν) → Normal(0,1)
    • ν  >  2  ⟹  Var = ν/(ν-2)  (apenas para ν>2)
    • Curtose excesso = 6/(ν-4)   (apenas para ν>4)
    • Caudas mais pesadas quanto menor ν
    • Evidência empírica: ν ≈ 3–6 para índices de ações
""")

# Ajuste t-Student (3 parâmetros: df, loc, scale)
df_t, loc_t, scale_t = stats.t.fit(r)
kurt_t = 6 / (df_t - 4) if df_t > 4 else float("inf")
print(f"  Ajuste ao IBOV:  ν={df_t:.2f}  μ={loc_t*100:.4f}%  "
      f"σ={scale_t*100:.4f}%  Curtose exc.={kurt_t:.2f}")
print(f"  Interpretação: ν={df_t:.1f} indica caudas {'moderadamente' if df_t>5 else 'muito'} pesadas")

print("\n" + "="*65)
print("  DISTRIBUIÇÃO DE PARETO / LEI DE POTÊNCIA  (cauda pesada)")
print("="*65)
print("""
  Definição de 'cauda pesada':
    P(|X| > x) ~ x^(-α)   para x → ∞  (α = índice de cauda)

  Comparação de decaimento:
    Normal:    P(|X|>x) ~ exp(-x²)   → decai exponencialmente (cauda leve)
    t(ν):      P(|X|>x) ~ x^(-ν)    → decai em lei de potência (cauda pesada)
    Pareto:    P(X  >x) = (x_m/x)^α  para x ≥ x_m

  Hill Estimator para α (estima o índice de cauda a partir dos dados):
    α̂ = k / Σᵢ₌₁ᵏ [ln X(n-i+1) - ln X(n-k)]
    onde X(n-k) são as k maiores observações (valores absolutos negativos)
""")

# Estimador de Hill para a cauda esquerda
losses = -r[r < 0]          # retornos negativos como perdas positivas
losses_sorted = np.sort(losses)[::-1]   # decrescente
k_range = range(10, 300, 5)
hill_estimates = []
for k in k_range:
    top_k = losses_sorted[:k]
    hill_est = k / np.sum(np.log(top_k) - np.log(top_k[-1]))
    hill_estimates.append(hill_est)
alpha_hill = np.mean(hill_estimates[-20:])   # média dos últimos pontos (estável)
print(f"  Hill estimator (cauda esquerda IBOV): α̂ ≈ {alpha_hill:.2f}")
print(f"  Como α̂ < 4, curtose teórica é infinita → risco extremo relevante!")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 2 – COMPARAÇÃO DE VaR ENTRE DISTRIBUIÇÕES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  VALUE AT RISK (VaR) — comparação entre distribuições")
print("="*65)
print("""
  VaR(α) = quantil α da distribuição de retornos
           = perda que não será excedida com probabilidade (1-α)

  Convenção: VaR 95% = 5% dos piores dias superam esse limiar
             VaR 99% = 1% dos piores dias superam esse limiar
""")

levels = [0.10, 0.05, 0.01, 0.005]
results = {}

for alpha in levels:
    var_hist  = np.percentile(r, alpha * 100)                     # Histórico / empírico
    var_norm  = stats.norm.ppf(alpha, mu_n, sig_n)                # Normal
    var_t     = stats.t.ppf(alpha, df_t, loc_t, scale_t)          # t-Student

    results[f"{int((1-alpha)*100)}%"] = {
        "VaR Histórico": f"{var_hist*100:.3f}%",
        "VaR Normal":    f"{var_norm*100:.3f}%",
        "VaR t-Student": f"{var_t*100:.3f}%",
        "Razão t/N":     f"{var_t/var_norm:.3f}×",
    }

df_var = pd.DataFrame(results).T
print(df_var.to_string())
print("""
  Leitura: "Razão t/N > 1" indica que a t-Student prevê perda maior
  que a Normal. Na prática, o VaR Normal subestima o risco de cauda,
  levando bancos e fundos a reservar capital insuficiente.
""")

# Violações do VaR 95% histórico
cutoff_95 = np.percentile(r, 5)
violations = np.sum(r < cutoff_95)
pct_viol = violations / len(r) * 100
print(f"  VaR 95% empírico = {cutoff_95*100:.3f}%")
print(f"  Violações observadas: {violations} dias = {pct_viol:.2f}% (esperado: 5,00%)")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 3 – VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.suptitle("S3 · Distribuições — IBOVESPA (log-retornos diários)",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

x_lin = np.linspace(r.min(), r.max(), 800)
x_tail = np.linspace(np.percentile(r, 0.1), np.percentile(r, 5), 300)

# --- Painel 1: PDF comparativa ---
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(r, bins=120, density=True, color="#BDBDBD", alpha=0.6, label="Dados obs.")
ax1.plot(x_lin, stats.norm.pdf(x_lin, mu_n, sig_n),
         "b-", lw=2.5, label=f"Normal(μ={mu_n*100:.3f}%, σ={sig_n*100:.3f}%)")
ax1.plot(x_lin, stats.t.pdf(x_lin, df_t, loc_t, scale_t),
         "r-", lw=2.5, label=f"t-Student(ν={df_t:.1f})")
ax1.set_xlim(-0.12, 0.12)
ax1.set_title("Ajuste de distribuições — região central", fontsize=11)
ax1.set_xlabel("Log-retorno diário")
ax1.set_ylabel("Densidade")
ax1.legend()

# --- Painel 2: Zoom na cauda esquerda ---
ax2 = fig.add_subplot(gs[0, 2])
x_zoom = np.linspace(r.min(), np.percentile(r, 3), 300)
hist_v, hist_e = np.histogram(r[r < np.percentile(r, 3)], bins=40, density=False)
# Normalizar para densidade
bin_w = hist_e[1] - hist_e[0]
ax2.bar(hist_e[:-1], hist_v / (len(r) * bin_w), width=bin_w,
        color="#E53935", alpha=0.5, label="Cauda obs.")
ax2.plot(x_zoom, stats.norm.pdf(x_zoom, mu_n, sig_n),
         "b-", lw=2, label="Normal")
ax2.plot(x_zoom, stats.t.pdf(x_zoom, df_t, loc_t, scale_t),
         "r-", lw=2, label="t-Student")
ax2.set_title("Zoom: cauda esquerda (piores 3%)", fontsize=11)
ax2.set_xlabel("Log-retorno")
ax2.legend(fontsize=8)

# --- Painel 3: Q-Q Normal ---
ax3 = fig.add_subplot(gs[1, 0])
(osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm", fit=True)
ax3.scatter(osm, osr, s=5, color="#5C6BC0", alpha=0.4)
ax3.plot(osm, slope*np.array(osm)+intercept, "k--", lw=1.5, label="Ref. Normal")
ax3.set_title("Q-Q Plot — Normal", fontsize=11)
ax3.set_xlabel("Quantis teóricos N(0,1)")
ax3.set_ylabel("Quantis observados")
ax3.legend(fontsize=8)

# --- Painel 4: Q-Q t-Student ---
ax4 = fig.add_subplot(gs[1, 1])
(osm2, osr2), (slope2, intercept2, _) = stats.probplot(r, dist="t", sparams=(df_t,), fit=True)
ax4.scatter(osm2, osr2, s=5, color="#E53935", alpha=0.4)
ax4.plot(osm2, slope2*np.array(osm2)+intercept2, "k--", lw=1.5, label=f"Ref. t({df_t:.1f})")
ax4.set_title("Q-Q Plot — t-Student", fontsize=11)
ax4.set_xlabel(f"Quantis teóricos t({df_t:.1f})")
ax4.set_ylabel("Quantis observados")
ax4.legend(fontsize=8)

# --- Painel 5: Hill estimator ---
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(list(k_range), hill_estimates, color="#7B2D8B", lw=2)
ax5.axhline(alpha_hill, color="red", ls="--", lw=1.5,
            label=f"α̂ ≈ {alpha_hill:.2f} (média estável)")
ax5.set_title("Hill Estimator — índice de cauda (α)", fontsize=11)
ax5.set_xlabel("k (número de extremos usados)")
ax5.set_ylabel("Estimativa de α")
ax5.legend(fontsize=8)

output_file = OUTPUT_DIR / "s3_02_distribuicoes.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
plt.show()
print(f"[✓] Gráfico salvo: {output_file}")
