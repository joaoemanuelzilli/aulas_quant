import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
ibov = precos["^BVSP"].dropna()
r = 100 * np.log(ibov).diff().dropna()
treino = r[:"2021"].to_numpy()             # 2010 a 2021
teste = r["2022":].to_numpy()              # 2022 a 2025
def rmse(previsao):
    return np.sqrt(np.mean((teste - previsao) ** 2))
media = treino.mean()
e2_media = (teste - media) ** 2            # erro² da ingênua
sinal = np.mean(np.sign(media) == np.sign(teste))
print(f"treino: {len(treino)} dias   teste: {len(teste)} dias")
print(f"previsão zero:    RMSE = {rmse(0.0):.4f}")
print(f"média do treino:  RMSE = {rmse(media):.4f}"
      f"   acerto de sinal = {sinal:.1%}")
for ordem in [(1, 0, 0), (2, 0, 2)]:
    m = ARIMA(treino, order=ordem).fit()
    # parâmetros fixos do treino; a cada dia entra o dado novo
    m = m.append(teste, refit=False)
    prev = m.predict(start=len(treino))    # um passo à frente
    acerto = np.mean(np.sign(prev) == np.sign(teste))
    # t da diferença média de erro² contra a média do treino
    d = (teste - prev) ** 2 - e2_media
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"ARIMA{ordem}:   RMSE = {rmse(prev):.4f}"
          f"   acerto de sinal = {acerto:.1%}   t = {t:.2f}")
# de onde vinha a autocorrelação do treino?
s = r[:"2021"]
sem2020 = pd.concat([s[:"2019"], s["2021":]])
print(f"autocorrelação de ordem 1 no treino: {s.autocorr(1):.3f}")
print(f"  só 2020: {s['2020'].autocorr(1):.3f}"
      f"   treino sem 2020: {sem2020.autocorr(1):.3f}"
      f"   teste: {r['2022':].autocorr(1):.3f}")
