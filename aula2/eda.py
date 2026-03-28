"""
═══════════════════════════════════════════════════════════════════════════════
  NÚCLEO QUANT — Liga de Mercado Financeiro · UFU · 2026.1
  Aula S2: Análise Exploratória de Ativos
  
  Tópicos:
    1. Coleta de dados com yfinance
    2. Tratamento de dados faltantes
    3. Visualização com Plotly
    4. Preços Normalizados (base 100)
    5. Retornos simples e log-retornos
    6. Volatilidade e histograma de retornos
    7. Correlação e heatmap
    8. Médias móveis (MM20 x MM200)
    9. Drawdown (Underwater chart) - extra
═══════════════════════════════════════════════════════════════════════════════
"""

# ─── Instalação (rode uma vez se necessário) ──────────────────────────────────
# pip install yfinance pandas numpy plotly

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# 1. COLETA DE DADOS COM yfinance
# ══════════════════════════════════════════════════════════════════════════════

import yfinance as yf

ACOES  = ["PETR3.SA", "VALE3.SA", "ITUB3.SA", "^BVSP"]
START  = "2019-01-01"
END    = "2026-03-26"

print("📥 Baixando dados...")

# Dólar baixado separadamente: calendário forex ≠ calendário B3
df_acoes = yf.download(ACOES, start=START, end=END, auto_adjust=True, progress=False)["Close"]
df_dolar  = yf.download("USDBRL=X", start=START, end=END, auto_adjust=False, progress=False)["Close"]
df_dolar  = df_dolar.squeeze()
df_dolar.name = "USDBRL=X"

# Normaliza índices para tz-naive (compatível com qualquer versão do yfinance)
def to_naive_date_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()

df_acoes.index = to_naive_date_index(df_acoes.index)
df_dolar.index = to_naive_date_index(df_dolar.index)

# Reindex pelo calendário B3 (mantendo NaN para tratar na etapa 2)
df_dolar = df_dolar.reindex(df_acoes.index)

# O yfinance rotula USDBRL=X pela ABERTURA da sessão forex (meia-noite UTC),
# enquanto a B3 fecha às ~21h UTC. O fechamento do Dólar no dia T do yfinance
# corresponde ao início do pregão B3 do dia T+1. Shift(-1) corrige esse offset.
df_dolar = df_dolar.shift(-1)

df = df_acoes.copy()
df["USDBRL=X"] = df_dolar

print(f"\n✅ Dados baixados: {df.shape[0]} dias x {df.shape[1]} ativos")
print(f"   Período: {df.index[0].date()} → {df.index[-1].date()}")
print(f"   NaN no Dólar após join: {df['USDBRL=X'].isna().sum()}")
print("\nPrimeiras linhas:")
print(df.head())

# ══════════════════════════════════════════════════════════════════════════════
# 2. DADOS FALTANTES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("🔍 DIAGNÓSTICO DE DADOS FALTANTES")
print("═"*60)

# Tratamento: preenche gaps internos e remove linhas sem dados válidos
df_clean = df.copy()
df_clean["USDBRL=X"] = df_clean["USDBRL=X"].ffill()
df_clean = df_clean.dropna()

print(f"\n✅ Após tratamento: {df_clean.shape[0]} dias úteis ({df.shape[0] - df_clean.shape[0]} linhas removidas)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. GRÁFICO DE PREÇOS NORMALIZADOS (base 100)
# ══════════════════════════════════════════════════════════════════════════════

# Normalizar para base 100 para comparar ativos em escalas diferentes

# Como comparar no mesmo gráfico um índice de 130.000 pontos (Ibovespa) com uma ação de R$30 (PETR3) e um Dólar a R$5?

df_norm = (df_clean / df_clean.iloc[0]) * 100

CORES = {
    "PETR3.SA": "#7B2FBE",
    "VALE3.SA": "#16A34A",
    "ITUB3.SA": "#2563EB",
    "USDBRL=X": "#D97706",
    "^BVSP":    "#DC2626",
}

NOMES = {
    "PETR3.SA": "PETR3",
    "VALE3.SA": "VALE3",
    "ITUB3.SA": "ITUB3",
    "USDBRL=X": "Dólar",
    "^BVSP":    "Ibovespa",
}

nome_petro = NOMES.get("PETR3.SA", "PETR3")

fig_precos = go.Figure()
for ticker in df_norm.columns:
    fig_precos.add_trace(go.Scatter(
        x=df_norm.index,
        y=df_norm[ticker],
        name=NOMES.get(ticker, ticker),
        line=dict(color=CORES.get(ticker, "#888888"), width=2),
        hovertemplate=f"<b>{NOMES.get(ticker, ticker)}</b><br>Data: %{{x|%d/%m/%Y}}<br>Base 100: %{{y:.1f}}<extra></extra>"
    ))

fig_precos.update_layout(
    title=dict(text="<b>Preços Normalizados (Base 100 = Jan/2019)</b>", font=dict(size=18)),
    xaxis_title="Data",
    yaxis_title="Índice (Base 100)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500,
)

# Adicionar linha de referência em 100
fig_precos.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5,
                     annotation_text="Base 100")

fig_precos.show()
print("\n📊 Gráfico 1: Preços normalizados")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RETORNOS: simples e log-retorno
# ══════════════════════════════════════════════════════════════════════════════

# Retorno simples (pct_change)
retornos = df_clean.pct_change().dropna()

# Log-retorno (preferido em modelos quantitativos)
log_ret = np.log(df_clean / df_clean.shift(1)).dropna()

print("\n" + "═"*60)
print("📈 ESTATÍSTICAS DOS RETORNOS DIÁRIOS")
print("═"*60)

stats = pd.DataFrame({
    "Retorno Médio Diário": retornos.mean(),
    "Retorno Médio Anual":  retornos.mean() * 252,
    "Vol Diária":           retornos.std(),
    "Vol Anual":            retornos.std() * np.sqrt(252),
    "Retorno Mín (pior dia)": retornos.min(),
    "Retorno Máx (melhor dia)": retornos.max(),
}).rename(index=NOMES)

stats = stats.map(lambda x: f"{x:.2%}")
print(stats.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. VOLATILIDADE: histogramas e volatilidade rolante
# ══════════════════════════════════════════════════════════════════════════════

# ── 5a. Histograma de retornos diários (PETR3 vs Dólar) ───────────────────
fig_hist = make_subplots(
    rows=1, cols=2,
    subplot_titles=[f"{nome_petro} — Retornos Diários", "Dólar — Retornos Diários"]
)

for col, (ticker, nome, cor) in enumerate([
    ("PETR3.SA", nome_petro, "#7B2FBE"),
    ("USDBRL=X", "Dólar",  "#D97706"),
], start=1):
    dados = retornos[ticker].dropna() * 100
    fig_hist.add_trace(
        go.Histogram(
            x=dados,
            nbinsx=80,
            name=nome,
            marker_color=cor,
            opacity=0.75,
            hovertemplate=f"Retorno: %{{x:.2f}}%<br>Freq: %{{y}}<extra>{nome}</extra>"
        ),
        row=1, col=col
    )
    # Linha vertical em zero
    fig_hist.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=col)

fig_hist.update_layout(
    title=dict(text="<b>Distribuição dos Retornos Diários — Cauda Gorda vs Cauda Fina</b>", font=dict(size=16)),
    showlegend=False,
    template="plotly_white",
    height=450,
)
fig_hist.update_xaxes(title_text="Retorno Diário (%)")
fig_hist.update_yaxes(title_text="Frequência")
fig_hist.show()
print("📊 Gráfico 2: Histograma de retornos")

# ── 5b. Volatilidade rolante (30 dias) ────────────────────────────────────
vol_anual = retornos.std() * np.sqrt(252)
print(f"\nVolatilidade Anual:")
for t, v in vol_anual.items():
    barra = "█" * int(v * 100)
    print(f"   {NOMES.get(t, t):10s} {v:.1%}  {barra}")

vol_rolante = retornos.rolling(30).std() * np.sqrt(252)

fig_vol = go.Figure()
for ticker in ["PETR3.SA", "VALE3.SA", "USDBRL=X"]:
    fig_vol.add_trace(go.Scatter(
        x=vol_rolante.index,
        y=vol_rolante[ticker],
        name=NOMES.get(ticker, ticker),
        line=dict(color=CORES.get(ticker), width=2),
        hovertemplate=f"<b>{NOMES.get(ticker)}</b><br>%{{x|%d/%m/%Y}}<br>Vol anualizada: %{{y:.1%}}<extra></extra>"
    ))

# Destaque: crash de março 2020
fig_vol.add_vrect(
    x0="2020-02-24", x1="2020-04-15",
    fillcolor="red", opacity=0.08, line_width=0,
    annotation_text="COVID-19 crash", annotation_position="top left"
)

fig_vol.update_layout(
    title=dict(text="<b>Volatilidade Rolante (30 dias, anualizada)</b>", font=dict(size=16)),
    xaxis_title="Data",
    yaxis_title="Volatilidade Anual",
    yaxis_tickformat=".0%",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=480,
)
fig_vol.show()
print("📊 Gráfico 3: Volatilidade rolante")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CORRELAÇÃO E HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

corr = retornos.rename(columns=NOMES).corr()

print("\n" + "═"*60)
print("🔗 MATRIZ DE CORRELAÇÃO")
print("═"*60)
print(corr.round(2).to_string())

# Heatmap interativo
fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="<b>Heatmap de Correlação — Retornos Diários (2019-2024)</b>",
    labels=dict(color="Correlação"),
    aspect="auto",
)
fig_corr.update_traces(
    hovertemplate="<b>%{x}</b> x <b>%{y}</b><br>Correlação: %{z:.2f}<extra></extra>"
)
fig_corr.update_layout(
    height=520,
    template="plotly_white",
    coloraxis_colorbar=dict(title="Correlação", tickvals=[-1, -0.5, 0, 0.5, 1]),
    title=dict(font=dict(size=16)),
)
fig_corr.show()
print("📊 Gráfico 4: Heatmap de correlação")

# ══════════════════════════════════════════════════════════════════════════════
# 7. MÉDIAS MÓVEIS (Golden Cross / Death Cross)
# ══════════════════════════════════════════════════════════════════════════════

close_petro = df_clean["PETR3.SA"]
mm20  = close_petro.rolling(20).mean()
mm200 = close_petro.rolling(200).mean()

# Identificar cruzamentos
sinal = (mm20 > mm200).astype(int)
cruzamentos = sinal.diff().dropna()
golden_cross = cruzamentos[cruzamentos == 1].index   # MM20 cruza MM200 para cima
death_cross  = cruzamentos[cruzamentos == -1].index  # MM20 cruza MM200 para baixo

fig_mm = go.Figure()

# Preço
fig_mm.add_trace(go.Scatter(
    x=close_petro.index, y=close_petro,
    name=f"{nome_petro} — Preço",
    line=dict(color="#AAAACC", width=1),
    opacity=0.8,
    hovertemplate=f"Preço: R$%{{y:.2f}}<br>%{{x|%d/%m/%Y}}<extra>{nome_petro}</extra>"
))

# MM20
fig_mm.add_trace(go.Scatter(
    x=mm20.index, y=mm20,
    name="MM20 (curta)",
    line=dict(color="#7B2FBE", width=2),
    hovertemplate="MM20: R$%{y:.2f}<extra></extra>"
))

# MM200
fig_mm.add_trace(go.Scatter(
    x=mm200.index, y=mm200,
    name="MM200 (longa)",
    line=dict(color="#16A34A", width=2.5),
    hovertemplate="MM200: R$%{y:.2f}<extra></extra>"
))

# Golden Cross (sinal de compra)
for idx in golden_cross:
    fig_mm.add_vline(
        x=idx.to_pydatetime(), line_dash="dash", line_color="#16A34A",
        line_width=1.5, opacity=0.6
    )

# Death Cross (sinal de venda)
for idx in death_cross:
    fig_mm.add_vline(
        x=idx.to_pydatetime(), line_dash="dash", line_color="#DC2626",
        line_width=1.5, opacity=0.6
    )

# Destaque crash COVID
fig_mm.add_vrect(
    x0="2020-02-24", x1="2020-04-15",
    fillcolor="red", opacity=0.06, line_width=0,
    annotation_text="COVID-19", annotation_position="top left",
    annotation_font=dict(size=10)
)

fig_mm.update_layout(
    title=dict(text=f"<b>{nome_petro} — Preço + MM20 x MM200 (Golden/Death Cross)</b>", font=dict(size=16)),
    xaxis_title="Data",
    yaxis_title="Preço (R$)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=520,
)
fig_mm.show()
print(f"📊 Gráfico 5: Médias móveis {nome_petro}")

print(f"\n   Golden Cross detectados: {len(golden_cross)}")
for d in golden_cross:
    print(f"     ↑ {d.date()}")
print(f"   Death Cross detectados:  {len(death_cross)}")
for d in death_cross:
    print(f"     ↓ {d.date()}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. DRAWDOWNS (Underwater Chart) - extra
# ══════════════════════════════════════════════════════════════════════════════

btc_raw = yf.download("BTC-USD", start=START, end=END, auto_adjust=True, progress=False)
btc_close = btc_raw["Close"] if "Close" in btc_raw else btc_raw
if isinstance(btc_close, pd.DataFrame):
    btc_close = btc_close.squeeze("columns")

btc_close.index = pd.to_datetime(btc_close.index)
btc_close = pd.to_numeric(btc_close, errors="coerce").ffill().dropna()


def calcular_drawdown(serie_preco: pd.Series) -> pd.Series:
    pico_acumulado = serie_preco.cummax()
    return (serie_preco / pico_acumulado) - 1


drawdown_PETR3 = calcular_drawdown(close_petro)
drawdown_btc = calcular_drawdown(btc_close)

pior_drawdown_PETR3 = drawdown_PETR3.min()
pior_drawdown_btc = drawdown_btc.min()

fig_underwater = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=[
        f"{nome_petro} — Pior drawdown: {pior_drawdown_PETR3:.1%}",
        f"BTC — Pior drawdown: {pior_drawdown_btc:.1%}",
    ],
)

fig_underwater.add_trace(
    go.Scatter(
        x=drawdown_PETR3.index,
        y=drawdown_PETR3,
        mode="lines",
        name=f"{nome_petro} Drawdown",
        line=dict(color="#B91C1C", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(220, 38, 38, 0.25)",
        hovertemplate=f"<b>{nome_petro}</b><br>%{{x|%d/%m/%Y}}<br>Drawdown: %{{y:.1%}}<extra></extra>",
        showlegend=False,
    ),
    row=1,
    col=1,
)

fig_underwater.add_trace(
    go.Scatter(
        x=drawdown_btc.index,
        y=drawdown_btc,
        mode="lines",
        name="BTC Drawdown",
        line=dict(color="#7F1D1D", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(185, 28, 28, 0.25)",
        hovertemplate="<b>BTC</b><br>%{x|%d/%m/%Y}<br>Drawdown: %{y:.1%}<extra></extra>",
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig_underwater.update_yaxes(tickformat=".0%", title_text="Drawdown", row=1, col=1)
fig_underwater.update_yaxes(tickformat=".0%", title_text="Drawdown", row=2, col=1)
fig_underwater.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7, row=1, col=1)
fig_underwater.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7, row=2, col=1)

fig_underwater.update_layout(
    title=dict(text="<b>Underwater Chart — Distância ao Pico Histórico</b>", font=dict(size=17)),
    template="plotly_white",
    height=700,
    hovermode="x unified",
)
fig_underwater.show()
print("📊 Gráfico 6: Drawdowns (Underwater)")
print(f"   {nome_petro} pior drawdown: {pior_drawdown_PETR3:.1%}")
print(f"   BTC   pior drawdown: {pior_drawdown_btc:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. DASHBOARD FINAL: tudo em um painel
# ══════════════════════════════════════════════════════════════════════════════

fig_dash = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Preços Normalizados (Base 100)",
        "Volatilidade Rolante 30 dias (anual)",
        f"Distribuição Retornos {nome_petro}",
        f"Médias Móveis {nome_petro}",
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.1,
)

# — Painel 1: Preços normalizados —
for ticker in df_norm.columns:
    fig_dash.add_trace(
        go.Scatter(x=df_norm.index, y=df_norm[ticker],
                   name=NOMES.get(ticker, ticker),
                   line=dict(color=CORES.get(ticker, "#888"), width=1.5),
                   showlegend=True),
        row=1, col=1
    )

# — Painel 2: Vol rolante —
for ticker in ["PETR3.SA", "VALE3.SA", "USDBRL=X"]:
    fig_dash.add_trace(
        go.Scatter(x=vol_rolante.index, y=vol_rolante[ticker],
                   name=NOMES.get(ticker),
                   line=dict(color=CORES.get(ticker), width=1.5),
                   showlegend=False),
        row=1, col=2
    )

# — Painel 3: Histograma —
fig_dash.add_trace(
    go.Histogram(x=retornos["PETR3.SA"].dropna() * 100,
                 nbinsx=70, marker_color="#7B2FBE", opacity=0.75,
                 showlegend=False),
    row=2, col=1
)

# — Painel 4: Médias móveis —
fig_dash.add_trace(go.Scatter(x=close_petro.index, y=close_petro,
    line=dict(color="#AAAACC", width=1), showlegend=False), row=2, col=2)
fig_dash.add_trace(go.Scatter(x=mm20.index, y=mm20,
    line=dict(color="#7B2FBE", width=2), name="MM20", showlegend=False), row=2, col=2)
fig_dash.add_trace(go.Scatter(x=mm200.index, y=mm200,
    line=dict(color="#16A34A", width=2.5), name="MM200", showlegend=False), row=2, col=2)

fig_dash.update_layout(
    title=dict(
        text="<b>Dashboard EDA — Núcleo Quant · UFU · 2026.1</b>",
        font=dict(size=18, color="#7B2FBE")
    ),
    template="plotly_white",
    height=750,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_dash.update_yaxes(tickformat=".0%", row=1, col=2)
fig_dash.update_xaxes(title_text="Retorno Diário (%)", row=2, col=1)
fig_dash.update_yaxes(title_text="Preço (R$)", row=2, col=2)

fig_dash.show()
print("📊 Gráfico 7: Dashboard completo")

# ══════════════════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("✅ ANÁLISE CONCLUÍDA — RESUMO")
print("═"*60)
print(f"\nAtivos analisados: {', '.join(NOMES.values())}")
print(f"Período:           {START} → {END}")
print(f"Dias úteis:        {df_clean.shape[0]}")
print(f"\nRetornos totais do período:")
retorno_total = (df_clean.iloc[-1] / df_clean.iloc[0] - 1)
for t, v in retorno_total.items():
    sinal_str = "📈" if v > 0 else "📉"
    print(f"  {sinal_str} {NOMES.get(t, t):10s}: {v:+.1%}")

print(f"\nAtivo mais volátil:  {NOMES.get(vol_anual.idxmax())} ({vol_anual.max():.1%} ao ano)")
print(f"Ativo mais estável:  {NOMES.get(vol_anual.idxmin())} ({vol_anual.min():.1%} ao ano)")
