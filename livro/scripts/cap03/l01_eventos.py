import pandas as pd

precos = pd.read_csv("../../dados/precos_ajustados.csv",
                     index_col=0, parse_dates=True)
p = precos[["PETR4.SA", "VALE3.SA"]].dropna()
R = p.pct_change().dropna().loc["2015":"2024"]
A = R["PETR4.SA"] > 0          # evento A: PETR4 sobe no dia
B = R["VALE3.SA"] > 0          # evento B: VALE3 sobe no dia

pA, pB, pAB = A.mean(), B.mean(), (A & B).mean()
pA_B, pA_nB = A[B].mean(), A[~B].mean()   # condicionais
print(f"{len(R)} pregões")
print(f"P(A) = {pA:.3f}    P(B) = {pB:.3f}")
print(f"P(A e B) = {pAB:.3f}    P(A) P(B) = {pA * pB:.3f}")
print(f"P(A | B) = {pA_B:.3f}    P(A | não B) = {pA_nB:.3f}")

# conferências: definição e lei da probabilidade total
assert abs(pA_B - pAB / pB) < 1e-12
assert abs(pA - (pA_B * pB + pA_nB * (1 - pB))) < 1e-12
