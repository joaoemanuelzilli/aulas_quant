"""Confere a saída e os números da Seção 'O sinal é significativo?'.

Roda l11_significancia.py, compara a saída com o bloco impresso em
capitulos/cap10.tex e refaz as contas citadas no texto. Termina com
erro se alguma deixar de valer.
"""
import re
import subprocess
import sys
from pathlib import Path

TEX = Path("../../capitulos/cap10.tex").read_text(encoding="utf8")
saida = subprocess.run([sys.executable, "l11_significancia.py"],
                       capture_output=True, text=True,
                       check=True).stdout.strip()
ini = TEX.index("\\label{sec:c10-significancia}")
bloco = re.search(r"\\begin\{lstlisting\}\[style=saida\]\n(.*?)"
                  r"\\end\{lstlisting\}", TEX[ini:], re.S).group(1)
assert bloco.strip() == saida, "saída impressa difere da execução"

num = [float(x) for x in re.findall(r"-?\d+\.\d+|\d+(?=%)", saida)]
sr, anos, t_iid, t_nw = num[0], num[1], num[2], num[3]
assert abs(sr * anos ** 0.5 - t_iid) < 0.02    # t = SR * raiz(n)
assert t_iid < 1.96 and t_nw < 1.96            # não significativo
texto = TEX[ini:ini + 6000]
for frase in ["$t$ de 0,45", "médio de 0,18", "66\\%",
              "3.462 deslocamentos",
              "62\\% na ação", "chega a 0,27"]:
    assert frase in texto, frase
print("conferência do Capítulo 10 (significância): ok")
