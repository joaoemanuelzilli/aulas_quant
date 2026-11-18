"""Confere a Seção 'Como ler um coeficiente' e a conta de Bayes.

Roda l08_coeficientes.py, compara a saída com o bloco impresso em
capitulos/cap12.tex e refaz as contas citadas no texto. Termina com
erro se alguma deixar de valer.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

TEX = Path("../../capitulos/cap12.tex").read_text(encoding="utf8")
saida = subprocess.run([sys.executable, "l08_coeficientes.py"],
                       capture_output=True, text=True,
                       check=True).stdout.strip()
ini = TEX.index("\\label{sec:c12-coeficientes}")
bloco = re.search(r"\\begin\{lstlisting\}\[style=saida\]\n(.*?)"
                  r"\\end\{lstlisting\}", TEX[ini:], re.S).group(1)
assert bloco.strip() == saida, "saída impressa difere da execução"

# números citados no texto da seção
texto = TEX[ini:TEX.index("\\subsection{A regressão preditiva")]
for frase in ["0,41\\%", "$-0,47$", "$-0,793$", "(0,65)", "(0,16)",
              "(1,11 e 0,41)", "$+0,0023$", "0,23\\%", "$t = 0,8$",
              "0,086 para 0,104", "(0,339)", "$+1,26$ e $-0,15$",
              "0,60, seis vezes o 0,10", "é 119, contra 1,4 sem",
              "\\sqrt{119/1,4} \\approx 9$", "$t \\approx 4,5$",
              "34\\%"]:
    assert frase in texto, frase
assert abs(0.65 / 0.16 - 4) < 0.1 and 1.109 / 0.408 < 3
assert abs(0.60 / 0.10 - 6) < 1e-9

# erro-padrão de MQO comum: razão = raiz da razão dos VIF
import contextlib, io  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    import l08_coeficientes as c  # noqa: E402
X2 = sm.add_constant(c.r[["ibov", "dolar"]])
X3 = sm.add_constant(c.r[["ibov", "bova", "dolar"]])
ep2 = sm.OLS(c.r["vale"], X2).fit().bse["ibov"]
ep3 = sm.OLS(c.r["vale"], X3).fit().bse["ibov"]
vif0 = 1 / (1 - sm.OLS(c.r["ibov"], sm.add_constant(
    c.r["dolar"])).fit().rsquared)
assert round(vif0, 1) == 1.4
assert np.isclose(ep3 / ep2, np.sqrt(c.vif / vif0), rtol=0.01)
assert round(ep3 / ep2) == 9 and round(np.sqrt(119 / 1.4)) == 9

# t^2 = (n-2) R^2 / (1 - R^2) em uma regressão simulada
rng = np.random.default_rng(0)
x = rng.normal(size=4000)
y = 0.07 * x + rng.normal(size=4000)
fit = sm.OLS(y, sm.add_constant(x)).fit()
r2 = fit.rsquared
assert np.isclose(fit.tvalues[1] ** 2, 3998 * r2 / (1 - r2))
assert abs(np.sqrt(3998 * 0.005 / 0.995) - 4.5) < 0.05

# precisão pelo teorema de Bayes, matriz de confusão do Código 12.3
vn, fp, fn, vp = 440, 391, 443, 468
base = (vp + fn) / 1742
recall, tfp = vp / (vp + fn), fp / (fp + vn)
prec = recall * base / (recall * base + tfp * (1 - base))
assert np.isclose(prec, vp / (vp + fp))
assert (round(recall, 3), round(base, 3), round(tfp, 3),
        round(prec, 3)) == (0.514, 0.523, 0.471, 0.545)
print("conferência do Capítulo 12 (coeficientes): ok")
