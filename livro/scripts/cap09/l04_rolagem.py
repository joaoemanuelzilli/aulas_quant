"""Retorno à vista, retorno de quem carrega o futuro e rolagem."""
import numpy as np
from futuros import fim_de_mes

for vista, fut in [("wti", "cl"), ("hh", "ng")]:
    m = fim_de_mes([vista, f"{fut}1", f"{fut}2"], fut)
    s, f1, f2 = m[vista], m[f"{fut}1"], m[f"{fut}2"]
    # compra o 2º contrato no fim do mês; um mês depois ele é o 1º
    r_fut = np.log(f1 / f2.shift())
    r_vista = np.log(s / s.shift())
    base = np.log(s / f1)               # base, em log
    rolagem = np.log(f1 / f2).shift()   # < 0 em contango
    soma = r_vista - base.diff() + rolagem
    assert np.allclose(r_fut.dropna(), soma.dropna())
    anos = r_fut.count() / 12
    print(f"{fut.upper()}, {r_fut.count()} meses,",
          "log-retorno ao ano:")
    for nome, x in [("à vista", r_vista), ("rolado", r_fut),
                    ("rolagem", rolagem), ("base", -base.diff())]:
        print(f"  {nome:8s} {x.sum() / anos:+7.1%}")
    # série contínua ajustada: termina no último preço e anda, a
    # cada mês, o retorno do contrato carregado
    acum = r_fut.fillna(0).cumsum()
    ajustada = f1.iloc[-1] * np.exp(acum - acum.iloc[-1])
    print(f"  US$ 1 vira: à vista {s.iloc[-1] / s.iloc[0]:.2f};",
          f"rolado {ajustada.iloc[-1] / ajustada.iloc[0]:.4f};",
          f"1º contrato emendado {f1.iloc[-1] / f1.iloc[0]:.2f}")
