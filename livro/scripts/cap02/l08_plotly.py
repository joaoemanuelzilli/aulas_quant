import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ohlcv = pd.read_csv("../../dados/petr4_ohlcv.csv",
                    index_col=0, parse_dates=True).loc["2024"]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75, 0.25],
                    vertical_spacing=0.03)
fig.add_trace(go.Candlestick(
    x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
    low=ohlcv["Low"], close=ohlcv["Close"], name="PETR4"),
    row=1, col=1)
fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["Volume"],
                     name="volume"), row=2, col=1)

# esconde os fins de semana, em que não há pregão
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
fig.update_layout(hovermode="x unified", showlegend=False,
                  xaxis_rangeslider_visible=False,
                  title="PETR4 em 2024 (preços ajustados)")

fig.write_html("petr4_2024.html", include_plotlyjs="cdn")
print(len(fig.data), "séries;", len(ohlcv), "pregões")
