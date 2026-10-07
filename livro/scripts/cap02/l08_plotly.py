import pandas as pd
import plotly.graph_objects as go

ohlcv = pd.read_csv("../../dados/petr4_ohlcv.csv",
                    index_col=0, parse_dates=True).loc["2024"]

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
    low=ohlcv["Low"], close=ohlcv["Close"], name="PETR4"))
fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["Volume"],
                     name="volume", yaxis="y2"))

# um eixo x para os dois painéis; cada eixo y ocupa uma faixa
fig.update_layout(
    yaxis=dict(domain=[0.30, 1.00]),         # preços, em cima
    yaxis2=dict(domain=[0.00, 0.24]),        # volume, embaixo
    xaxis=dict(anchor="y2", rangeslider_visible=False,
               rangebreaks=[dict(bounds=["sat", "mon"])]),
    hovermode="x unified", hoversubplots="axis",
    showlegend=False, title="PETR4 em 2024 (preços ajustados)")

fig.write_html("petr4_2024.html", include_plotlyjs="cdn")
print(len(fig.data), "séries;", len(ohlcv), "pregões")
