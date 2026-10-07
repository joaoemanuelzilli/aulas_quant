"""
═══════════════════════════════════════════════════════════════════════════════
  NÚCLEO QUANT — Liga de Mercado Financeiro · UFU · 2026.1
  Aula: Otimização de Portfólio — Teoria de Markowitz

  Tópicos:
    1. Retorno esperado e risco de um portfólio
    2. A matriz de covariância — o coração do modelo
    3. Simulação de Monte Carlo — nuvem de portfólios aleatórios
    4. Fronteira Eficiente — a curva ótima
    5. Portfólio de Mínima Variância (MVP)
    6. Portfólio de Máximo Sharpe Ratio (MSR)
    7. Linha do Mercado de Capitais (CML)
    8. Dashboard final
═══════════════════════════════════════════════════════════════════════════════
"""

# ─── Instalação (rode uma vez se necessário) ──────────────────────────────────
# pip install yfinance pandas numpy plotly scipy

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# ══════════════════════════════════════════════════════════════════════════════
# 0. COLETA DE DADOS (igual à Aula S2 — reutilizando o pipeline)
# ══════════════════════════════════════════════════════════════════════════════

import yfinance as yf

ACOES = ["PETR3.SA", "VALE3.SA", "ITUB3.SA", "BBAS3.SA", "WEGE3.SA"]
START = "2020-01-01"
END   = "2026-03-26"
DIAS_UTEIS_ANO = 252

NOMES = {
    "PETR3.SA": "PETR3",
    "VALE3.SA":  "VALE3",
    "ITUB3.SA":  "ITUB3",
    "BBAS3.SA":  "BBAS3",
    "WEGE3.SA":  "WEGE3",
}

CORES = {
    "PETR3.SA": "#7B2FBE",
    "VALE3.SA":  "#16A34A",
    "ITUB3.SA":  "#2563EB",
    "BBAS3.SA":  "#D97706",
    "WEGE3.SA":  "#DC2626",
}

print("📥 Baixando dados...")
df_raw = yf.download(ACOES, start=START, end=END, auto_adjust=True, progress=False)["Close"]

def to_naive_date_index(idx):
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()

df_raw.index = to_naive_date_index(df_raw.index)
df_clean = df_raw.ffill().dropna()

print(f"✅ {df_clean.shape[0]} dias úteis | {df_clean.shape[1]} ativos")
print(f"   Período: {df_clean.index[0].date()} → {df_clean.index[-1].date()}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. RETORNO ESPERADO E RISCO DE UM PORTFÓLIO
# ══════════════════════════════════════════════════════════════════════════════
#
# Para um portfólio com N ativos e pesos w = [w1, w2, ..., wN]:
#
#   Retorno esperado:   E[Rp] = Σ wi * E[Ri]   →  produto vetorial: w · μ
#   Variância:          σ²p   = wᵀ · Σ · w      →  forma quadrática com a matriz de covariância
#   Desvio padrão:      σp    = √(wᵀ · Σ · w)
#
# A CHAVE é a matriz de covariância Σ (sigma):
#   - Captura como CADA par de ativos se move junto
#   - É ela que torna a diversificação possível matematicamente
# ══════════════════════════════════════════════════════════════════════════════

# Log-retornos diários (preferido em modelos quant — propriedade aditiva)
log_ret = np.log(df_clean / df_clean.shift(1)).dropna()

# Vetor de retornos médios anuais (μ)
mu = log_ret.mean() * DIAS_UTEIS_ANO

# Matriz de covariância anualizada (Σ)
cov = log_ret.cov() * DIAS_UTEIS_ANO

print("\n" + "═"*60)
print("📐 MATRIZ DE COVARIÂNCIA ANUALIZADA (Σ)")
print("═"*60)
print(cov.rename(index=NOMES, columns=NOMES).round(4).to_string())

print("\n" + "═"*60)
print("📈 RETORNOS ESPERADOS ANUAIS (μ)")
print("═"*60)
for ticker, ret in mu.items():
    barra = "█" * int(abs(ret) * 30)
    sinal = "📈" if ret > 0 else "📉"
    print(f"   {sinal} {NOMES[ticker]:8s}  {ret:+.1%}  {barra}")

# ── Funções auxiliares ────────────────────────────────────────────────────────

def retorno_portfolio(pesos, mu):
    """Retorno esperado anual: produto escalar pesos · μ"""
    return np.dot(pesos, mu)

def risco_portfolio(pesos, cov):
    """Volatilidade anual: √(wᵀ Σ w)"""
    return np.sqrt(np.dot(pesos, np.dot(cov, pesos)))

def sharpe_ratio(pesos, mu, cov, rf=0.1075):
    """
    Sharpe Ratio = (Retorno - Taxa livre de risco) / Volatilidade
    rf = 0.1075 ≈ Selic atual (~10.75% ao ano)
    """
    r = retorno_portfolio(pesos, mu)
    s = risco_portfolio(pesos, cov)
    return (r - rf) / s

N = len(ACOES)

# Exemplo: portfólio igualmente ponderado
pesos_iguais = np.array([1/N] * N)
print(f"\n📊 Portfólio igualmente ponderado (1/{N} cada):")
print(f"   Retorno esperado: {retorno_portfolio(pesos_iguais, mu):.2%}")
print(f"   Volatilidade:     {risco_portfolio(pesos_iguais, cov):.2%}")
print(f"   Sharpe Ratio:     {sharpe_ratio(pesos_iguais, mu, cov):.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. A MATRIZ DE COVARIÂNCIA — visualização pedagógica
# ══════════════════════════════════════════════════════════════════════════════
#
# Diferença entre Covariância e Correlação:
#   Covariância: magnitude depende das escalas dos ativos (difícil de interpretar)
#   Correlação:  normalizada entre -1 e +1 (fácil de interpretar)
#   Correlação(A,B) = Cov(A,B) / (σA * σB)
#
# Por que usamos covariância e não correlação no Markowitz?
#   Porque a fórmula da variância do portfólio (wᵀ Σ w) precisa
#   da escala real das volatilidades — correlação perde essa informação.
# ══════════════════════════════════════════════════════════════════════════════

corr = log_ret.corr()
nomes_lista = [NOMES[t] for t in ACOES]

fig_cov = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Matriz de Covariância (Σ)", "Matriz de Correlação"],
    horizontal_spacing=0.15,
)

# Heatmap covariância
fig_cov.add_trace(go.Heatmap(
    z=cov.values,
    x=nomes_lista, y=nomes_lista,
    colorscale="Blues",
    text=[[f"{v:.4f}" for v in row] for row in cov.values],
    texttemplate="%{text}",
    showscale=True,
    colorbar=dict(x=0.44, title="Cov"),
    hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Cov: %{z:.4f}<extra></extra>",
), row=1, col=1)

# Heatmap correlação
fig_cov.add_trace(go.Heatmap(
    z=corr.values,
    x=nomes_lista, y=nomes_lista,
    colorscale="RdBu_r",
    zmin=-1, zmax=1,
    text=[[f"{v:.2f}" for v in row] for row in corr.values],
    texttemplate="%{text}",
    showscale=True,
    colorbar=dict(x=1.0, title="Corr"),
    hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Corr: %{z:.2f}<extra></extra>",
), row=1, col=2)

fig_cov.update_layout(
    title=dict(text="<b>Covariância vs Correlação — mesma informação, escalas diferentes</b>", font=dict(size=16)),
    template="plotly_white",
    height=420,
)
fig_cov.show()
print("\n📊 Gráfico 1: Covariância vs Correlação")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SIMULAÇÃO DE MONTE CARLO — nuvem de portfólios aleatórios
# ══════════════════════════════════════════════════════════════════════════════
#
# Ideia: gerar N_SIM portfólios com pesos aleatórios e plotar cada um
#        no espaço Risco (eixo X) × Retorno (eixo Y).
#
# O que emerge dessa nuvem:
#   - A borda ESQUERDA da nuvem → Fronteira Eficiente
#   - O ponto mais à ESQUERDA   → Portfólio de Mínima Variância
#   - O ponto de maior INCLINAÇÃO da reta da origem → Máximo Sharpe
# ══════════════════════════════════════════════════════════════════════════════

N_SIM = 10_000
RF    = 0.1075  # Taxa Selic (~10.75% ao ano)

np.random.seed(42)

sim_retornos  = np.zeros(N_SIM)
sim_riscos    = np.zeros(N_SIM)
sim_sharpes   = np.zeros(N_SIM)
sim_pesos     = np.zeros((N_SIM, N))

for i in range(N_SIM):
    # Pesos aleatórios que somam 1 (sem short selling)
    w = np.random.dirichlet(np.ones(N))   # Dirichlet é o jeito correto: já soma 1 e ≥ 0
    sim_pesos[i]    = w
    sim_retornos[i] = retorno_portfolio(w, mu)
    sim_riscos[i]   = risco_portfolio(w, cov)
    sim_sharpes[i]  = sharpe_ratio(w, mu, cov, RF)

print(f"\n✅ {N_SIM:,} portfólios simulados")
print(f"   Retorno range: {sim_retornos.min():.1%} → {sim_retornos.max():.1%}")
print(f"   Risco range:   {sim_riscos.min():.1%} → {sim_riscos.max():.1%}")
print(f"   Sharpe range:  {sim_sharpes.min():.2f} → {sim_sharpes.max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. OTIMIZAÇÃO — Fronteira Eficiente, MVP e MSR
# ══════════════════════════════════════════════════════════════════════════════
#
# Markowitz formulou isso como um problema de otimização:
#
#   DADO um nível de retorno alvo r*:
#     minimizar   wᵀ Σ w          (minimizar risco)
#     sujeito a:  w · μ = r*      (atingir retorno alvo)
#                 Σ wi  = 1       (pesos somam 100%)
#                 wi   ≥ 0        (sem venda a descoberto)
#
# Resolvemos isso para vários valores de r* → traçamos a Fronteira Eficiente
# ══════════════════════════════════════════════════════════════════════════════

# ── Restrições e bounds comuns ────────────────────────────────────────────────
bounds      = tuple((0, 1) for _ in range(N))          # 0 ≤ wi ≤ 1
restricao_soma = {"type": "eq", "fun": lambda w: np.sum(w) - 1}   # Σwi = 1
pesos_ini   = np.array([1/N] * N)                       # chute inicial: pesos iguais

# ── 4a. Portfólio de Mínima Variância (MVP) ──────────────────────────────────
resultado_mvp = minimize(
    fun=risco_portfolio,
    x0=pesos_ini,
    args=(cov,),
    method="SLSQP",
    bounds=bounds,
    constraints=restricao_soma,
)
pesos_mvp   = resultado_mvp.x
ret_mvp     = retorno_portfolio(pesos_mvp, mu)
risco_mvp   = risco_portfolio(pesos_mvp, cov)
sharpe_mvp  = sharpe_ratio(pesos_mvp, mu, cov, RF)

print("\n" + "═"*60)
print("🔵 PORTFÓLIO DE MÍNIMA VARIÂNCIA (MVP)")
print("═"*60)
print(f"   Retorno: {ret_mvp:.2%}  |  Risco: {risco_mvp:.2%}  |  Sharpe: {sharpe_mvp:.2f}")
print("   Pesos:")
for ticker, peso in zip(ACOES, pesos_mvp):
    barra = "█" * int(peso * 40)
    print(f"     {NOMES[ticker]:8s}  {peso:.1%}  {barra}")

# ── 4b. Portfólio de Máximo Sharpe Ratio (MSR) ───────────────────────────────
resultado_msr = minimize(
    fun=lambda w: -sharpe_ratio(w, mu, cov, RF),   # negativo porque minimize minimiza
    x0=pesos_ini,
    method="SLSQP",
    bounds=bounds,
    constraints=restricao_soma,
)
pesos_msr   = resultado_msr.x
ret_msr     = retorno_portfolio(pesos_msr, mu)
risco_msr   = risco_portfolio(pesos_msr, cov)
sharpe_msr  = sharpe_ratio(pesos_msr, mu, cov, RF)

print("\n" + "═"*60)
print("⭐ PORTFÓLIO DE MÁXIMO SHARPE RATIO (MSR)")
print("═"*60)
print(f"   Retorno: {ret_msr:.2%}  |  Risco: {risco_msr:.2%}  |  Sharpe: {sharpe_msr:.2f}")
print("   Pesos:")
for ticker, peso in zip(ACOES, pesos_msr):
    barra = "█" * int(peso * 40)
    print(f"     {NOMES[ticker]:8s}  {peso:.1%}  {barra}")

# ── 4c. Fronteira Eficiente ───────────────────────────────────────────────────
# Varre retornos-alvo entre o retorno do MVP e o maior retorno individual
# Para cada alvo, minimiza o risco com a restrição de retorno

alvos = np.linspace(ret_mvp, mu.max(), 80)

front_riscos   = []
front_retornos = []

for alvo in alvos:
    restricoes = [
        restricao_soma,
        {"type": "eq", "fun": lambda w, a=alvo: retorno_portfolio(w, mu) - a},
    ]
    res = minimize(
        fun=risco_portfolio,
        x0=pesos_ini,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=restricoes,
    )
    if res.success:
        front_riscos.append(risco_portfolio(res.x, cov))
        front_retornos.append(retorno_portfolio(res.x, mu))

print(f"\n✅ Fronteira eficiente: {len(front_retornos)} pontos calculados")

# ── 4d. Linha do Mercado de Capitais (CML) ───────────────────────────────────
# CML: reta que parte da taxa livre de risco (RF) e é tangente à fronteira
# O ponto de tangência é exatamente o portfólio de Máximo Sharpe (MSR)
# Equação: E[Rp] = RF + Sharpe_MSR * σp

risco_cml  = np.linspace(0, max(front_riscos) * 1.1, 100)
retorno_cml = RF + sharpe_msr * risco_cml

# ══════════════════════════════════════════════════════════════════════════════
# 5. GRÁFICO PRINCIPAL — Fronteira Eficiente completa
# ══════════════════════════════════════════════════════════════════════════════

fig_front = go.Figure()

# Nuvem de Monte Carlo (colorida pelo Sharpe)
fig_front.add_trace(go.Scatter(
    x=sim_riscos,
    y=sim_retornos,
    mode="markers",
    marker=dict(
        color=sim_sharpes,
        colorscale="Viridis",
        size=3,
        opacity=0.5,
        colorbar=dict(title="Sharpe Ratio", thickness=15, x=1.02),
        showscale=True,
    ),
    name="Portfólios simulados",
    hovertemplate="Risco: %{x:.1%}<br>Retorno: %{y:.1%}<extra>Monte Carlo</extra>",
))

# Fronteira eficiente
fig_front.add_trace(go.Scatter(
    x=front_riscos,
    y=front_retornos,
    mode="lines",
    line=dict(color="#1E293B", width=3),
    name="Fronteira Eficiente",
    hovertemplate="Risco: %{x:.1%}<br>Retorno: %{y:.1%}<extra>Fronteira Eficiente</extra>",
))

# Linha do Mercado de Capitais (CML)
fig_front.add_trace(go.Scatter(
    x=risco_cml,
    y=retorno_cml,
    mode="lines",
    line=dict(color="#F59E0B", width=2, dash="dash"),
    name=f"CML (inclinação = Sharpe MSR = {sharpe_msr:.2f})",
    hovertemplate="Risco: %{x:.1%}<br>CML: %{y:.1%}<extra>CML</extra>",
))

# Portfólio de Mínima Variância (MVP)
fig_front.add_trace(go.Scatter(
    x=[risco_mvp], y=[ret_mvp],
    mode="markers+text",
    marker=dict(color="#2563EB", size=16, symbol="diamond"),
    text=["MVP"], textposition="top right",
    name=f"Mín. Variância  |  Sharpe {sharpe_mvp:.2f}",
    hovertemplate=f"<b>MVP</b><br>Risco: {risco_mvp:.1%}<br>Retorno: {ret_mvp:.1%}<br>Sharpe: {sharpe_mvp:.2f}<extra></extra>",
))

# Portfólio de Máximo Sharpe (MSR)
fig_front.add_trace(go.Scatter(
    x=[risco_msr], y=[ret_msr],
    mode="markers+text",
    marker=dict(color="#DC2626", size=16, symbol="star"),
    text=["MSR"], textposition="top right",
    name=f"Máx. Sharpe  |  Sharpe {sharpe_msr:.2f}",
    hovertemplate=f"<b>MSR</b><br>Risco: {risco_msr:.1%}<br>Retorno: {ret_msr:.1%}<br>Sharpe: {sharpe_msr:.2f}<extra></extra>",
))

# Ativos individuais
for ticker in ACOES:
    r_i = mu[ticker]
    s_i = np.sqrt(cov.loc[ticker, ticker])
    fig_front.add_trace(go.Scatter(
        x=[s_i], y=[r_i],
        mode="markers+text",
        marker=dict(color=CORES[ticker], size=12, symbol="circle",
                    line=dict(color="white", width=1.5)),
        text=[NOMES[ticker]], textposition="middle right",
        name=NOMES[ticker],
        hovertemplate=f"<b>{NOMES[ticker]}</b><br>Risco: {s_i:.1%}<br>Retorno: {r_i:.1%}<extra></extra>",
    ))

# Ponto da taxa livre de risco
fig_front.add_trace(go.Scatter(
    x=[0], y=[RF],
    mode="markers+text",
    marker=dict(color="#6B7280", size=10, symbol="square"),
    text=["Selic (RF)"], textposition="middle right",
    name=f"Taxa livre de risco ({RF:.1%})",
    hovertemplate=f"Selic: {RF:.1%}<extra></extra>",
))

fig_front.update_layout(
    title=dict(
        text="<b>Fronteira Eficiente de Markowitz — Portfólio de 5 Ativos B3</b>",
        font=dict(size=18)
    ),
    xaxis=dict(title="Risco (Volatilidade Anual)", tickformat=".0%"),
    yaxis=dict(title="Retorno Esperado Anual", tickformat=".0%"),
    template="plotly_white",
    legend=dict(orientation="v", x=1.08, y=0.5, xanchor="left"),
    hovermode="closest",
    height=600,
    width=1100,
)
fig_front.show()
print("📊 Gráfico 2: Fronteira Eficiente completa")

# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPOSIÇÃO DOS PORTFÓLIOS ÓTIMOS — gráfico de barras
# ══════════════════════════════════════════════════════════════════════════════

portfolios = {
    f"Igualmente\nPonderado\n(1/{N} cada)": pesos_iguais,
    f"Mín. Variância\n(MVP)":               pesos_mvp,
    f"Máx. Sharpe\n(MSR)":                  pesos_msr,
}

fig_comp = go.Figure()

for ticker, cor in zip(ACOES, CORES.values()):
    fig_comp.add_trace(go.Bar(
        name=NOMES[ticker],
        x=list(portfolios.keys()),
        y=[portfolios[p][list(ACOES).index(ticker)] for p in portfolios],
        marker_color=cor,
        hovertemplate=f"<b>{NOMES[ticker]}</b><br>Peso: %{{y:.1%}}<extra></extra>",
    ))

fig_comp.update_layout(
    title=dict(text="<b>Composição dos Portfólios Ótimos</b>", font=dict(size=16)),
    barmode="stack",
    yaxis=dict(title="Peso no Portfólio", tickformat=".0%"),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=450,
)
fig_comp.show()
print("📊 Gráfico 3: Composição dos portfólios")

# ══════════════════════════════════════════════════════════════════════════════
# 7. BACKTEST SIMPLES — como cada portfólio teria performado?
# ══════════════════════════════════════════════════════════════════════════════
#
# ATENÇÃO: isso é in-sample — usamos os MESMOS dados para otimizar e testar.
# Em aplicações reais, separe treino e teste (walk-forward, expanding window).
# Aqui o objetivo é pedagógico: mostrar o efeito da diversificação.
# ══════════════════════════════════════════════════════════════════════════════

retornos_diarios = log_ret.copy()

# Retorno diário de cada portfólio = soma ponderada dos retornos diários
ret_port_igual = retornos_diarios.dot(pesos_iguais)
ret_port_mvp   = retornos_diarios.dot(pesos_mvp)
ret_port_msr   = retornos_diarios.dot(pesos_msr)

# Curva de patrimônio acumulada (base 100)
patrimonio = pd.DataFrame({
    f"Igualmente Ponderado": np.exp(ret_port_igual.cumsum()) * 100,
    f"Mín. Variância (MVP)": np.exp(ret_port_mvp.cumsum()) * 100,
    f"Máx. Sharpe (MSR)":    np.exp(ret_port_msr.cumsum()) * 100,
})

# Adicionar ativos individuais para comparação
for ticker in ACOES:
    patrimonio[NOMES[ticker]] = np.exp(retornos_diarios[ticker].cumsum()) * 100

fig_bt = go.Figure()

# Portfólios otimizados (linhas grossas)
estilos_port = [
    ("Igualmente Ponderado", "#6B7280", 2.5, "solid"),
    ("Mín. Variância (MVP)", "#2563EB", 3,   "solid"),
    ("Máx. Sharpe (MSR)",    "#DC2626", 3,   "solid"),
]
for nome, cor, lw, dash in estilos_port:
    fig_bt.add_trace(go.Scatter(
        x=patrimonio.index, y=patrimonio[nome],
        name=nome,
        line=dict(color=cor, width=lw, dash=dash),
        hovertemplate=f"<b>{nome}</b><br>%{{x|%d/%m/%Y}}<br>Base 100: %{{y:.1f}}<extra></extra>",
    ))

# Ativos individuais (linhas finas, tracejadas)
for ticker in ACOES:
    fig_bt.add_trace(go.Scatter(
        x=patrimonio.index, y=patrimonio[NOMES[ticker]],
        name=NOMES[ticker],
        line=dict(color=CORES[ticker], width=1.2, dash="dot"),
        opacity=0.6,
        hovertemplate=f"<b>{NOMES[ticker]}</b><br>%{{x|%d/%m/%Y}}<br>Base 100: %{{y:.1f}}<extra></extra>",
    ))

fig_bt.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4,
                 annotation_text="Base 100")

fig_bt.update_layout(
    title=dict(text="<b>Backtest In-Sample — Curva de Patrimônio (Base 100 = Jan/2020)</b>",
               font=dict(size=16)),
    xaxis_title="Data",
    yaxis_title="Patrimônio (Base 100)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=520,
)
fig_bt.show()
print("📊 Gráfico 4: Backtest in-sample")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TABELA DE MÉTRICAS FINAIS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_drawdown(serie):
    pico = serie.cummax()
    return (serie / pico) - 1

metricas = {}
for nome_col, ret_serie in [
    (f"Igualmente Ponderado", ret_port_igual),
    (f"Mín. Variância (MVP)", ret_port_mvp),
    (f"Máx. Sharpe (MSR)",    ret_port_msr),
]:
    r_anual   = ret_serie.mean() * DIAS_UTEIS_ANO
    vol_anual = ret_serie.std() * np.sqrt(DIAS_UTEIS_ANO)
    sr        = (r_anual - RF) / vol_anual
    pat       = np.exp(ret_serie.cumsum()) * 100
    dd        = calcular_drawdown(pat).min()
    metricas[nome_col] = {
        "Retorno Anual": f"{r_anual:.2%}",
        "Volatilidade":  f"{vol_anual:.2%}",
        "Sharpe Ratio":  f"{sr:.2f}",
        "Pior Drawdown": f"{dd:.1%}",
    }

df_metricas = pd.DataFrame(metricas).T
print("\n" + "═"*60)
print("📋 TABELA DE MÉTRICAS — PORTFÓLIOS OTIMIZADOS")
print("═"*60)
print(df_metricas.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 9. DASHBOARD FINAL
# ══════════════════════════════════════════════════════════════════════════════

fig_dash = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Fronteira Eficiente",
        "Composição dos Portfólios",
        "Curva de Patrimônio (Base 100)",
        "Sharpe Ratio por Portfólio",
    ],
    vertical_spacing=0.16,
    horizontal_spacing=0.12,
)

# — Painel 1: Fronteira Eficiente (simplificada) —
fig_dash.add_trace(go.Scatter(
    x=sim_riscos, y=sim_retornos,
    mode="markers",
    marker=dict(color=sim_sharpes, colorscale="Viridis", size=2, opacity=0.4),
    showlegend=False,
), row=1, col=1)

fig_dash.add_trace(go.Scatter(
    x=front_riscos, y=front_retornos,
    mode="lines", line=dict(color="#1E293B", width=2),
    name="Fronteira", showlegend=False,
), row=1, col=1)

fig_dash.add_trace(go.Scatter(
    x=[risco_mvp, risco_msr], y=[ret_mvp, ret_msr],
    mode="markers+text",
    marker=dict(color=["#2563EB", "#DC2626"], size=12, symbol=["diamond", "star"]),
    text=["MVP", "MSR"], textposition="top right",
    showlegend=False,
), row=1, col=1)

# — Painel 2: Composição (barras empilhadas) —
for i, (ticker, cor) in enumerate(CORES.items()):
    fig_dash.add_trace(go.Bar(
        name=NOMES[ticker],
        x=list(portfolios.keys()),
        y=[portfolios[p][i] for p in portfolios],
        marker_color=cor,
        showlegend=True,
    ), row=1, col=2)

# — Painel 3: Patrimônio acumulado —
for nome_col, cor, lw in [
    ("Igualmente Ponderado", "#6B7280", 1.5),
    ("Mín. Variância (MVP)", "#2563EB", 2),
    ("Máx. Sharpe (MSR)",    "#DC2626", 2),
]:
    fig_dash.add_trace(go.Scatter(
        x=patrimonio.index, y=patrimonio[nome_col],
        name=nome_col, line=dict(color=cor, width=lw),
        showlegend=False,
    ), row=2, col=1)

# — Painel 4: Sharpe Ratio comparativo —
sharpes_plot = {
    "Igualmente\nPonderado": sharpe_ratio(pesos_iguais, mu, cov, RF),
    "MVP":                   sharpe_mvp,
    "MSR":                   sharpe_msr,
}
fig_dash.add_trace(go.Bar(
    x=list(sharpes_plot.keys()),
    y=list(sharpes_plot.values()),
    marker_color=["#6B7280", "#2563EB", "#DC2626"],
    text=[f"{v:.2f}" for v in sharpes_plot.values()],
    textposition="outside",
    showlegend=False,
), row=2, col=2)

fig_dash.update_layout(
    title=dict(
        text="<b>Dashboard Markowitz — Núcleo Quant · UFU · 2026.1</b>",
        font=dict(size=18, color="#1E293B"),
    ),
    template="plotly_white",
    height=750,
    barmode="stack",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_dash.update_xaxes(tickformat=".0%", row=1, col=1)
fig_dash.update_yaxes(tickformat=".0%", row=1, col=1)
fig_dash.update_yaxes(tickformat=".0%", row=1, col=2)
fig_dash.update_yaxes(title_text="Base 100", row=2, col=1)
fig_dash.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)

fig_dash.show()
print("📊 Gráfico 5: Dashboard completo")

# ══════════════════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("✅ ANÁLISE CONCLUÍDA — RESUMO MARKOWITZ")
print("═"*60)
print(f"\nAtivos: {', '.join(NOMES.values())}")
print(f"Período: {START} → {END} | {df_clean.shape[0]} dias úteis")
print(f"Taxa livre de risco (Selic): {RF:.2%}")

print(f"""
┌─────────────────────────┬──────────┬────────────┬────────────┐
│ Portfólio               │ Retorno  │ Volatil.   │ Sharpe     │
├─────────────────────────┼──────────┼────────────┼────────────┤
│ Igualmente Ponderado    │ {retorno_portfolio(pesos_iguais, mu):>7.2%}  │ {risco_portfolio(pesos_iguais, cov):>9.2%}  │ {sharpe_ratio(pesos_iguais, mu, cov, RF):>9.2f}  │
│ Mín. Variância (MVP)    │ {ret_mvp:>7.2%}  │ {risco_mvp:>9.2%}  │ {sharpe_mvp:>9.2f}  │
│ Máx. Sharpe (MSR)       │ {ret_msr:>7.2%}  │ {risco_msr:>9.2%}  │ {sharpe_msr:>9.2f}  │
└─────────────────────────┴──────────┴────────────┴────────────┘
""")

print("💡 LIMITAÇÕES DO MODELO (para discussão):")
print("   1. Retornos históricos ≠ retornos futuros (garbage in, garbage out)")
print("   2. Otimização é extremamente sensível ao vetor μ estimado")
print("   3. Backtest in-sample superestima performance real")
print("   4. Correlações não são estáveis — mudam em crises")
print("   5. Sem custos de transação, impostos ou liquidez")
print("   6. Próximo passo natural: Black-Litterman, Risk Parity, Rebalanceamento")