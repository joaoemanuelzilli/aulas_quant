"""Confere o Capítulo 9 contra a execução dos scripts.

Rode de dentro de livro/scripts/cap09:  python conf_c09.py
1. cada bloco style=saida de capitulos/cap09.tex é a saída literal
   do script impresso logo antes dele, e nenhuma linha de código
   impresso passa de 66 caracteres;
2. os números citados no texto, nas legendas e nos exercícios saem
   das contas abaixo (termina com erro se algum deixar de valer).
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from futuros import fim_de_mes, ler
from l06_sazonal import spread_por_mes, vol_por_mes, vol_por_prazo

TEX = Path("../../capitulos/cap09.tex").read_text(encoding="utf8")
PADRAO = re.compile(
    r"\\lstinputlisting\[[^\]]*\]\{scripts/cap09/(\w+\.py)\}\s*"
    r"\\begin\{lstlisting\}\[style=saida\]\n(.*?)\\end\{lstlisting\}",
    re.S)


def limpo(texto):
    return "\n".join(l.rstrip() for l in texto.strip("\n").splitlines())


blocos = PADRAO.findall(TEX)
assert len(blocos) == 6, f"esperava 6 pares código/saída: {len(blocos)}"
for script, colado in blocos:
    linhas = Path(script).read_text(encoding="utf8").splitlines()
    longas = [l for l in linhas if len(l) > 66]
    assert not longas, f"{script}: linha longa: {longas[0]}"
    r = subprocess.run([sys.executable, script], capture_output=True,
                       text=True)
    assert r.returncode == 0, f"{script} falhou:\n{r.stderr}"
    assert limpo(colado) == limpo(r.stdout), f"{script}: saída difere"
    print(f"ok: saída de {script}")


def perto(x, alvo, tol):
    assert abs(x - alvo) <= tol, (x, alvo)


# --- abertura e Seção 9.4: gás de 2008 a 2024
m = fim_de_mes(["hh", "ng1", "ng2"], "ng")
perto(1 - m.hh.iloc[-1] / m.hh.iloc[0], 0.78, 0.005)          # 78%
r = np.log(m.ng1 / m.ng2.shift()).sum()
assert 1 - np.exp(r) > 0.99                                    # >99%
perto(1 - np.exp(r), 0.997, 0.0005)                            # 99,7%
perto((m.ng2 > m.ng1).mean(), 0.81, 0.005)                     # 81%
mc = fim_de_mes(["wti", "cl1", "cl2"], "cl")
perto((mc.cl2 > mc.cl1).mean(), 0.68, 0.005)                   # 68%
tb = pd.read_csv("../../dados/c09_tbill.csv", index_col=0,
                 parse_dates=True)["tbill"]
perto(tb.loc["2008":"2024-03"].mean(), 0.98, 0.005)            # 0,98%
for d, v in [("2020-04-21", 0.11), ("2022-03-08", 0.36)]:
    perto(tb[d], v, 1e-9)

# --- Seção 9.1: ajuste do contrato de junho de 2020
perto((11.57 / 20.43 - 1), -0.434, 0.001)                      # 43%
perto((27.40 - 16.94) * 1000, 10460, 1e-6)

# --- Seção 9.2: custo de carregamento e futuros financeiros
perto(80 * np.exp(0.07 * 0.5), 82.85, 0.005)
perto(85 - 80 * np.exp(0.07 * 0.5), 2.15, 0.005)
perto(0.07 - np.log(79 / 80) / 0.5, 0.095, 0.0005)
perto(5.40 * np.exp(0.05), 5.68, 0.005)

# --- Seção 9.3: curva
f = ler(["cl1", "cl2", "cl3", "cl4"], "1990").dropna()
s = f.cl4 - f.cl1
assert len(f) == 8033
perto((s > 0).mean(), 0.525, 0.0005)
perto(s.max(), 66.14, 1e-9)
d = ler(["cl1", "cl4"]).loc["2020-04-21"]
assert d.cl4 / d.cl1 > 2
b = ler(["brent"]).brent
perto(b.loc["2022"].max(), 133.18, 1e-9)
assert b.loc["2022"].idxmax() == pd.Timestamp("2022-03-08")
hh = ler(["hh"]).hh.dropna()
n = hh.groupby(hh.index.year).size()
assert n.loc[1997:2006].min() == 58 and n.loc[1997:2006].max() == 143
assert n.loc[2008:2025].min() > 236

# --- Seção 9.5: hedge e base Brent - WTI
p = ler(["wti", "brent", "hh"], "2008").dropna()   # como a figura
p = p[(p > 0).all(axis=1)]
dif = p.brent - p.wti
perto(dif["2011":"2013"].mean(), 14.84, 0.005)
perto(dif[:"2010"].mean(), -0.94, 0.005)
perto(dif["2025"].mean(), 3.57, 0.005)
perto(dif["2026-09"].mean(), 17.31, 0.005)
perto(dif.max(), 30.78, 1e-9)
assert dif.idxmax() == pd.Timestamp("2026-09-25")
perto(1.03 * 1_000_000 / 130_000, 7.9, 0.05)                    # índice
# h* do Brent só em 2011-2013 e no treino sem esses anos
mh = fim_de_mes(["wti", "brent", "cl1", "cl2"], "cl")
rh = pd.DataFrame({"s": mh.brent.pct_change(),
                   "f": mh.cl1 / mh.cl2.shift() - 1}).dropna()
hb = lambda z: np.cov(z.s, z.f)[0, 1] / np.var(z.f, ddof=1)
perto(hb(rh.loc["2011":"2013"]), 0.60, 0.005)
tr = rh.loc["2008":"2015"]
perto(hb(tr.drop(tr.loc["2011":"2013"].index)), 0.95, 0.005)

# --- Seção 9.10: calendário e erro-padrão da correlação defasada
cal = ler(["wti", "brent"], "2008")
so_b = cal.index[cal.wti.isna() & cal.brent.notna()]   # falta WTI
so_w = cal.index[cal.brent.isna() & cal.wti.notna()]   # falta Brent
for d in ["2025-07-04", "2025-11-27"]:
    assert pd.Timestamp(d) in so_b                       # feriados EUA
for d in ["2025-04-21", "2025-12-26"]:
    assert pd.Timestamp(d) in so_w                       # feriados RU
anos = pd.Series(so_b.year).value_counts()
assert anos.idxmax() == 2011
perto(1 / np.sqrt(4596), 0.015, 0.0005)

# --- Seção 9.8: sazonalidade e efeito Samuelson
sp, vol = spread_por_mes(), vol_por_mes()
for mes, v in [(9, 0.056), (10, 0.073), (1, -0.025)]:
    perto(sp[mes], v, 0.0005)
perto(vol[[7, 8]].mean(), 0.39, 0.005)
perto(vol[[11, 12, 1]].mean(), 0.70, 0.005)
for fut, alvo in [("cl", [0.405, 0.367, 0.342]),
                  ("ng", [0.466, 0.430, 0.389])]:
    for v, a in zip(vol_por_prazo(fut), alvo):
        perto(v, a, 0.0005)
lr = np.log(hh.loc["2008":][hh > 0]).diff().abs()
assert set(lr[lr > 1].index.strftime("%Y-%m")) == {
    "2021-02", "2024-01", "2026-01"}

# --- Seção 9.10: correlação móvel WTI x Brent
q = ler(["wti", "brent", "hh"], "2008").dropna()
q = q[(q > 0).all(axis=1)]
w = np.log(q.resample("W-FRI").last().dropna()).diff().dropna()
rc = w.wti.rolling(52).corr(w.brent)
perto(rc.min(), 0.52, 0.005)
perto(rc.max(), 0.98, 0.005)

# --- Seção 9.9: sensibilidade ao início do teste
import statsmodels.api as sm
from l07_previsao import previsoes
for ini in ["2010-01", "2012-01", "2017-01"]:
    for hz in (2, 3):
        pv = previsoes(hz, ini)
        ev = pv.drop(columns="real").sub(pv["real"], axis=0)
        razao_f = (ev.futuro ** 2).mean() / (ev.passeio ** 2).mean()
        assert 0.855 <= razao_f <= 0.915, (ini, hz, razao_f)
        dv = ev.futuro ** 2 - ev.passeio ** 2
        tv = sm.OLS(dv, np.ones(len(dv))).fit(
            cov_type="HAC", cov_kwds={"maxlags": hz}).tvalues.iloc[0]
        assert -1.65 <= tv <= -1.25, (ini, hz, tv)

# --- gabaritos numéricos dos exercícios
aj = np.diff([130_000, 130_500, 129_200, 131_000]) * 2
assert list(aj) == [1000, -2600, 3600] and aj.sum() == 2000    # 9.1
F = 2000 * np.exp(0.04 * 0.25)
perto(F, 2020.10, 0.005); perto(2035 - F, 14.90, 0.005)        # 9.2
perto(5.40 * np.exp((0.14 - 0.04) * 0.5), 5.6769, 0.00005)     # 9.3
h = 0.9 * 0.30 / 0.35
perto(h, 0.7714, 0.00005); perto(0.9 ** 2, 0.81, 1e-12)        # 9.4
perto(h * 500_000 / 1000, 385.7, 0.05)
perto(1.2 * 2e6 / 125_000, 19.2, 1e-9)                         # 9.5
perto((1.2 - 0.5) * 2e6 / 125_000, 11.2, 1e-9)
print("conf_c09: todos os números conferem")
