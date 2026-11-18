"""Testes das funções centrais do Capítulo 9 (rode desta pasta)."""
import numpy as np
import pandas as pd
from futuros import fim_de_mes, ler, mes_do_contrato, vencimento
from l05_hedge import razao, reducao

uteis = ler(["cl1"]).index
ng = ler(["ng1"]).index

# 1. vencimentos conhecidos: a EIA informa que o último pregão do
#    contrato de maio de 2020 foi 21/04/2020
assert vencimento(pd.Period("2020-05"), uteis, "cl") == \
    pd.Timestamp("2020-04-21")
assert vencimento(pd.Period("2020-06"), uteis, "cl") == \
    pd.Timestamp("2020-05-19")
# 25/01/2020 foi sábado: recua ao dia útil anterior e conta três
assert vencimento(pd.Period("2020-03"), uteis, "cl") == \
    pd.Timestamp("2020-02-20")
# gás: três dias úteis antes de 01/03/2016 (29, 26 e 25 de fevereiro)
assert vencimento(pd.Period("2016-03"), ng, "ng") == \
    pd.Timestamp("2016-02-25")

# 2. o contrato 1 muda no pregão seguinte ao vencimento
t1, t2 = pd.Timestamp("2020-04-21"), pd.Timestamp("2020-04-22")
assert mes_do_contrato(t1, 1, uteis, "cl") == pd.Period("2020-05")
assert mes_do_contrato(t2, 1, uteis, "cl") == pd.Period("2020-06")
assert mes_do_contrato(t1, 2, uteis, "cl") == pd.Period("2020-06")

# 3. fins de mês: depois do vencimento e sem meses faltando
for cols, prod in [(["wti", "cl1", "cl2"], "cl"),
                   (["hh", "ng1", "ng2"], "ng")]:
    m = fim_de_mes(cols, prod)
    for t in m.index:
        assert vencimento(t.to_period("M") + 1,
                          ler([f"{prod}1"]).index, prod) < t
    assert len(m) == 196

# 4. razão de hedge = coeficiente angular de MQO e minimiza a
#    variância da posição protegida
rng = np.random.default_rng(9)
f = rng.normal(0, 0.1, 5000)
s = 0.8 * f + rng.normal(0, 0.05, 5000)
h = razao(s, f)
assert np.isclose(h, np.polyfit(f, s, 1)[0])
assert abs(h - 0.8) < 0.01
assert all(reducao(s, f, h) >= reducao(s, f, x) - 1e-12
           for x in np.linspace(0, 1.6, 33))

# 5. rolagem em uma curva parada: à vista constante, contrato 2
#    1% acima do 1; quem carrega o futuro perde ln(1,01) por mês
f1, f2 = 50 * 1.01, 50 * 1.01 ** 2
r_fut = np.log(f1 / f2)              # compra f2, um mês depois é f1
assert np.isclose(r_fut, -np.log(1.01))

# 6. custo de carregamento: F = S exp((r + u - y) T)
S, r, u, T = 80.0, 0.05, 0.02, 0.5
F = S * np.exp((r + u) * T)
assert abs(F - 82.85) < 0.005
y = r + u - np.log(79 / S) / T       # conveniência implícita
assert abs(y - 0.0951) < 0.0001
print("teste_c09: tudo certo")
