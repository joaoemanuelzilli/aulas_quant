"""Features e alvo a partir do OHLCV diário de PETR4."""
import numpy as np
import pandas as pd


def carregar():
    return pd.read_csv("../../dados/petr4_ohlcv.csv",
                       index_col=0, parse_dates=True)


def wilder(x, n):
    """Média de Wilder: começa na média simples dos n primeiros
    valores e depois segue m_t = m_{t-1} + (x_t - m_{t-1}) / n."""
    m = x.copy()
    m.iloc[:n - 1] = np.nan
    m.iloc[n - 1] = x.iloc[:n].mean()
    return m.ewm(alpha=1 / n, adjust=False).mean()


def rsi(p, n=14):
    d = p.diff().dropna()
    ganho = wilder(d.clip(lower=0), n)       # altas médias
    perda = wilder(-d.clip(upper=0), n)      # quedas médias
    return 100 - 100 / (1 + ganho / perda)


def mme(x, n):
    """Média móvel exponencial com alfa = 2 / (n + 1)."""
    return x.ewm(span=n, adjust=False).mean()


def montar(d):
    """Cada linha t usa apenas dados até o fechamento de t."""
    p, v = d["Close"], d["Volume"]
    r = np.log(p).diff()
    X = pd.DataFrame(index=d.index)
    X["ret_1"] = r                           # log-retorno do dia
    X["ret_5"] = r.rolling(5).sum()          # momentum semanal
    X["ret_21"] = r.rolling(21).sum()        # momentum mensal
    X["vol_21"] = r.rolling(21).std()        # vol. realizada
    X["p_mm50"] = p / p.rolling(50).mean() - 1
    X["rsi_14"] = rsi(p, 14)
    macd = (mme(p, 12) - mme(p, 26)) / p     # em fração do preço
    X["macd_h"] = macd - mme(macd, 9)        # histograma
    mm, dp = p.rolling(20).mean(), p.rolling(20).std(ddof=0)
    X["pct_b"] = (p - (mm - 2 * dp)) / (4 * dp)
    X["vol_rel"] = np.log(v / v.rolling(20).mean())
    # t é sexta: o alvo de 1 dia é, em geral, o retorno de segunda
    X["sexta"] = (d.index.dayofweek == 4).astype(float)
    return X


def base(h=1):
    """Features em t e log-retorno de t até t+h, alinhados."""
    d = carregar()
    X = montar(d)
    p = np.log(d["Close"])
    y = (p.shift(-h) - p).rename("alvo")
    dados = X.join(y).dropna()
    return dados.drop(columns="alvo"), dados["alvo"]


if __name__ == "__main__":
    d = carregar()
    X, y = base(h=1)
    print(f"pregões no arquivo: {len(d)}")
    print(f"linhas completas: {len(X)}, de "
          f"{X.index[0].date()} a {X.index[-1].date()}")
    # regra de ouro: refazer a conta sem o futuro dá o mesmo
    rng = np.random.default_rng(0)
    cortes = rng.choice(np.arange(300, len(d)), 40, replace=False)
    cheio = montar(d)
    igual = all(np.allclose(montar(d.iloc[:i + 1]).iloc[-1],
                            cheio.iloc[i]) for i in cortes)
    print("40 datas sorteadas iguais sem os dados futuros:",
          igual)
    c = X.corr().abs()
    c = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1))
    print(c.stack().nlargest(3).round(2).to_string())
    # três linhas da tabela de treino, usadas na Seção 9.1
    ex = X.loc["2024-03-11":"2024-03-13",
               ["ret_1", "ret_5", "vol_21", "rsi_14"]].copy()
    ex["alvo"] = y
    ex["rotulo"] = (y > 0).astype(int)
    print(ex.round(4).to_string())
