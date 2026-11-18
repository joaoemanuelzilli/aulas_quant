"""Testes das identidades usadas em 'Como ler um coeficiente'."""
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(1)
n = 3000
x2 = rng.normal(size=n)
x1 = -0.8 * x2 + rng.normal(size=n)
y = 1.1 * x1 + 0.4 * x2 + rng.normal(size=n)


def coef(y, *xs):
    return sm.OLS(y, sm.add_constant(np.column_stack(xs))).fit()


def teste_vies_de_variavel_omitida_e_exato():
    b = coef(y, x1, x2).params
    delta = coef(x1, x2).params[1]
    assert np.isclose(coef(y, x2).params[1], b[2] + b[1] * delta)


def teste_padronizado_simples_e_correlacao():
    b = coef(y, x1).params[1]
    assert np.isclose(b * x1.std() / y.std(),
                      np.corrcoef(x1, y)[0, 1])


def teste_binaria_sozinha_e_diferenca_de_medias():
    d = (rng.random(n) < 0.3).astype(float)
    b = coef(y, d).params[1]
    assert np.isclose(b, y[d == 1].mean() - y[d == 0].mean())


def teste_vif_infla_variancia():
    x3 = x1 + 0.1 * rng.normal(size=n)
    vif = 1 / (1 - coef(x1, x3, x2).rsquared)
    sozinho = coef(y, x1, x2).bse[1]
    junto = coef(y, x1, x3, x2).bse[1]
    # razão dos erros-padrão ~ raiz da razão dos VIF
    vif0 = 1 / (1 - coef(x1, x2).rsquared)
    assert abs(junto / sozinho / np.sqrt(vif / vif0) - 1) < 0.05


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("teste_"):
            f()
            print("ok", nome)
