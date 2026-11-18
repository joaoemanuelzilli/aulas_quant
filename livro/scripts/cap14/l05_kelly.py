import numpy as np

p, b = 0.55, 1.0          # prob. de ganhar; ganho por real
q = 1 - p
f_kelly = p - q / b


def crescimento(f):
    """Crescimento esperado do log da riqueza por aposta."""
    return p * np.log(1 + b * f) + q * np.log(1 - f)


if __name__ == "__main__":
    print(f"f* = {f_kelly:.2f}")
    rng = np.random.default_rng(11)
    ganhou = rng.random((10_000, 500)) < p   # 10 mil caminhos
    for mult in (0.5, 1.0, 2.0):
        f = mult * f_kelly
        fator = np.where(ganhou, 1 + b * f, 1 - f)
        riqueza = fator.cumprod(axis=1)
        pico = np.maximum.accumulate(riqueza, axis=1)
        queda = (1 - riqueza / pico).max(axis=1)
        final = riqueza[:, -1]
        print(f"f = {f:.2f} ({mult:.1f} x Kelly): "
              f"g = {crescimento(f):+.5f}  "
              f"mediana final {np.median(final):5.2f}  "
              f"queda {np.median(queda):.0%}")
