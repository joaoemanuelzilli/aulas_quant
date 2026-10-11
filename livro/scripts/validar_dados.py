"""Valida os arquivos gerados por limpar_dados.py.

Refaz as contas por um caminho independente do script de
limpeza (lê o extrato da B3 e a tabela de eventos por conta
própria), compara com os arquivos anteriores, que vinham só do
Yahoo, imprime um relatório e termina com erro se alguma
conferência obrigatória falhar.

Pastas (variáveis de ambiente, com padrão):
  DADOS_LIVRO    arquivos novos        (padrão: ../dados)
  DADOS_ANTIGOS  arquivos anteriores   (padrão: dados/ do livro)
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
NOVO = Path(os.environ.get("DADOS_LIVRO", AQUI.parent / "dados"))
LIVRO = next((p for p in AQUI.parents
              if (p / "livro.tex").exists()), AQUI.parent)
ANTIGO = Path(os.environ.get("DADOS_ANTIGOS", LIVRO / "dados"))
DATA_CISAO, FATOR_CISAO = pd.Timestamp("2021-10-04"), 1 - 0.1754
PAPEIS = ["PETR4", "VALE3", "ITUB4", "WEGE3", "ABEV3", "BBAS3",
          "MGLU3", "BOVA11"]
GERADOS = ["precos_sem_ajuste.csv", "precos_ajustados.csv",
           "petr4_ohlcv.csv", "volume_financeiro.csv"]
CITADAS = {
    "PETR4": ["2011-01-03", "2011-05-11", "2011-05-12",
              "2011-05-13", "2011-08-17", "2014-04-02",
              "2017-05-18", "2019-01-02", "2020-03-09",
              "2021-02-22", "2021-02-23", "2024-03-08",
              "2024-03-11"],
    "BBAS3": ["2016-03-15"],
    "ITUB4": ["2013-05-20", "2013-05-21", "2021-10-04"],
    "VALE3": ["2019-01-28"],
}
# Diferenças de retorno anual acima de 1 p.p. já examinadas
# (item 3). Uma diferença fora desta lista derruba o validador.
_ITUB = ("fechamento do Yahoo em 30/12/2013, último pregão do ano, "
         "2,3% acima do oficial (32,07 contra 31,35 na escala da "
         "B3); o erro sai de 2013 e entra em 2014")
_MGLU = ("o Yahoo trata a subscrição de 01/02/2024 como "
         "desdobramento (fator 0,9353 sobre todo o passado); no "
         "fechamento da B3 não há degrau nesse dia")
EXPLICADAS = {("ITUB4", "2013"): _ITUB, ("ITUB4", "2014"): _ITUB,
              ("MGLU3", "2024"): _MGLU, ("MGLU3", "2010-25"): _MGLU}

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 30)
falhas = []


def titulo(texto):
    print("\n" + "=" * 70 + "\n" + texto + "\n" + "=" * 70)


def conferir(condicao, mensagem):
    """Registra o resultado de uma conferência obrigatória."""
    print(("  OK     " if condicao else "  FALHA  ") + mensagem)
    if not condicao:
        falhas.append(mensagem)


def ler(pasta, nome):
    return pd.read_csv(pasta / nome, index_col=0, parse_dates=True,
                       float_precision="round_trip")


def md5(caminho):
    return hashlib.md5(Path(caminho).read_bytes()).hexdigest()


def ret(serie):
    """Retorno entre cotações válidas consecutivas."""
    return serie.dropna().pct_change().dropna()


# ---------------------------------------------------------------
# leitura
# ---------------------------------------------------------------
b3 = pd.read_csv(NOVO / "bruto" / "b3_cotahist.csv",
                 parse_dates=["data"])
b3["codigo"] = b3["codigo"].replace({"AMBV3": "ABEV3"})
eventos = pd.read_csv(NOVO / "bruto" / "eventos_societarios.csv",
                      parse_dates=["data_ex"])
eventos["f"] = [float(Fraction(f)) for f in eventos["fator"]]
correcoes = pd.read_csv(NOVO / "bruto" / "correcoes_proventos.csv",
                        parse_dates=["data_ex"])
fech = b3.pivot(index="data", columns="codigo",
                values="fechamento")
qtd = b3.pivot(index="data", columns="codigo",
               values="quantidade")
pregoes = fech.index

sem = ler(NOVO, "precos_sem_ajuste.csv")
aj = ler(NOVO, "precos_ajustados.csv")
sem_ant = ler(ANTIGO, "precos_sem_ajuste.csv")
aj_ant = ler(ANTIGO, "precos_ajustados.csv")
y_sem = ler(NOVO / "bruto", "precos_sem_ajuste.csv")
y_aj = ler(NOVO / "bruto", "precos_ajustados.csv")

print("arquivos novos:    ", NOVO)
print("arquivos anteriores:", ANTIGO)
if NOVO.resolve() == ANTIGO.resolve():
    print("AVISO: as duas pastas são a mesma; as comparações "
          "antes/depois (itens 3, 5 e 6) ficam vazias.")

# ---------------------------------------------------------------
titulo("0. Formato: mesmas colunas e mesmo índice dos arquivos "
       "anteriores")
for nome in ["precos_sem_ajuste.csv", "precos_ajustados.csv"]:
    n, a = ler(NOVO, nome), ler(ANTIGO, nome)
    conferir(list(n.columns) == list(a.columns),
             f"{nome}: colunas iguais e na mesma ordem")
    conferir(a.index.difference(n.index).empty,
             f"{nome}: todas as {len(a)} datas anteriores presentes")
    conferir(pregoes.difference(n.index).empty,
             f"{nome}: todos os {len(pregoes)} pregões da B3 "
             f"presentes ({len(n)} linhas)")
    outras = ["^GSPC", "BRL=X"]
    y = ler(NOVO / "bruto", nome).reindex(n.index)
    conferir(n[outras].equals(y[outras]),
             f"{nome}: ^GSPC e BRL=X idênticos ao bruto do Yahoo")
    ib = y["^BVSP"].where(n.index.isin(pregoes))
    conferir(n["^BVSP"].equals(ib),
             f"{nome}: ^BVSP igual ao do Yahoo nos pregões da B3 e "
             "ausente fora deles")
    conferir(n["^BVSP"].equals(a["^BVSP"].reindex(n.index)),
             f"{nome}: ^BVSP idêntico ao do arquivo anterior")
o_novo, o_ant = ler(NOVO, "petr4_ohlcv.csv"), ler(ANTIGO,
                                                 "petr4_ohlcv.csv")
conferir(list(o_novo.columns) == list(o_ant.columns),
         "petr4_ohlcv.csv: colunas iguais e na mesma ordem "
         f"{list(o_novo.columns)}")
for nome in ["volume_financeiro.csv"]:
    t = ler(NOVO, nome)
    conferir(t.index.equals(sem.index) and list(t.columns)
             == [c for c in sem.columns if c.endswith(".SA")],
             f"{nome}: mesmo índice e mesmas colunas .SA")

# ---------------------------------------------------------------
titulo("1. Retorno do arquivo sem ajuste = retorno do fechamento "
       "bruto da B3")
print("Em data de evento a comparação é feita depois de aplicar "
      "o fator da tabela;\nem ITUB4 04/10/2021, o fator da cisão. "
      "Tolerância: 1e-8.\n")
linhas = []
for p in PAPEIS:
    bruto = fech[p].dropna()
    fator = pd.Series(1.0, index=bruto.index)
    for e in eventos[eventos["papel"] == p].itertuples():
        fator[e.data_ex] = e.f
    if p == "ITUB4":
        fator[DATA_CISAO] = FATOR_CISAO
    esperado = (bruto / (bruto.shift(1) * fator) - 1).dropna()
    novo = sem[p + ".SA"].reindex(bruto.index)
    obtido = (novo / novo.shift(1) - 1).dropna()
    erro = (obtido - esperado).abs()
    n_ev = int((fator != 1).sum())
    linhas.append([p, len(esperado), n_ev, erro.max(),
                   int((erro > 1e-8).sum())])
    conferir(len(obtido) == len(esperado) and erro.max() < 1e-8,
             f"{p}: {len(esperado)} retornos, {n_ev} com fator, "
             f"erro máximo {erro.max():.1e}")
    conferir(abs(novo.iloc[-1] - bruto.iloc[-1]) < 1e-9,
             f"{p}: último valor ({novo.iloc[-1]:.2f}) é o "
             "fechamento negociado na B3")
print("\ntotal de retornos conferidos:",
      sum(linha[1] for linha in linhas))

print("\nTabela de eventos usada, com o salto observado no "
      "fechamento da B3 na data ex:")
tab = []
for e in eventos.itertuples():
    s = fech[e.papel].dropna()
    i = s.index.get_loc(e.data_ex)
    tab.append([e.papel, e.data_ex.date(), e.fator, round(e.f, 4),
                s.iloc[i - 1], s.iloc[i],
                round(s.iloc[i] / s.iloc[i - 1], 4),
                f"{s.iloc[i] / (s.iloc[i - 1] * e.f) - 1:+.2%}"])
print(pd.DataFrame(tab, columns=[
    "papel", "data_ex", "fator", "decimal", "fech_véspera",
    "fech_dia", "razão_B3", "retorno_ajustado"]).to_string(
        index=False))
conferir(all(abs(float(t[7].rstrip("%"))) < 12 for t in tab),
         "todo evento explica o salto do dia (retorno ajustado "
         "abaixo de 12% em módulo)")

print("\nFator de proventos (ajustado / sem ajuste):")
print("Fora das datas da tabela de correções, o degrau do fator "
      "tem de ser o do Yahoo.\n")
passos = {}
for p in PAPEIS:
    c = p + ".SA"
    dias = fech[p].dropna().index
    f = (aj[c] / sem[c]).reindex(dias)
    y = (y_aj[c] / y_sem[c]).reindex(dias).ffill()
    passo, passo_y = f / f.shift(1), y / y.shift(1)
    passos[p] = (passo, passo_y)
    dcor = pd.DatetimeIndex(
        correcoes.loc[correcoes["papel"] == p, "data_ex"])
    fora = ~dias.isin(dcor)
    dif = (passo / passo_y - 1).abs()[fora].max()
    n_ex = int((passo > 1 + 1e-4).sum())
    conferir(passo.min() > 1 - 1e-9 and dif < 5e-5,
             f"{p}: não decrescente; {n_ex} datas ex; "
             f"{len(dcor)} corrigidas; nas demais, desvio máximo do "
             f"degrau em relação ao Yahoo {dif:.1e}; "
             f"fator de {f.iloc[0]:.4f} a {f.iloc[-1]:.4f}")

print("\nCorreções de proventos aplicadas "
      "(bruto/correcoes_proventos.csv).")
print("Valores por ação na base da época; retorno ajustado do dia "
      "ex antes (fator do Yahoo)\ne depois da correção.\n")
tab = []
for e in correcoes.itertuples():
    p, d = e.papel, e.data_ex
    s = fech[p].dropna()
    vesp = s.shift(1)[d]
    k = eventos[(eventos["papel"] == p)
                & (eventos["data_ex"] == d)]["f"].prod()
    r_sem = s[d] / (vesp * k) - 1
    passo, passo_y = passos[p]
    impl_y = (1 - 1 / passo_y[d]) * vesp
    impl_n = (1 - 1 / passo[d]) * vesp
    tab.append([p, d.date(), vesp, e.valor_yahoo, impl_y,
                e.valor_oficial, impl_n, r_sem,
                (1 + r_sem) * passo_y[d] - 1,
                (1 + r_sem) * passo[d] - 1])
tab = pd.DataFrame(tab, columns=[
    "papel", "data_ex", "fech_véspera", "yahoo_tabela",
    "yahoo_implícito", "oficial", "novo_implícito", "ret_sem",
    "ret_aj_antes", "ret_aj_depois"])
pc = "{:+.2%}".format
print(tab.to_string(index=False, formatters={
    "yahoo_implícito": "{:.4f}".format,
    "novo_implícito": "{:.4f}".format, "ret_sem": pc,
    "ret_aj_antes": pc, "ret_aj_depois": pc}))
conferir(np.allclose(tab["novo_implícito"], tab["oficial"],
                     atol=1e-6),
         "em toda correção, o provento implícito no fator novo é "
         "o valor oficial")
conferir(np.allclose(tab["yahoo_implícito"], tab["yahoo_tabela"],
                     atol=6e-5),
         "o valor do Yahoo anotado na tabela é o implícito no "
         "fator do Yahoo")
for p in PAPEIS:
    c = p + ".SA"
    r_aj = aj[c].dropna().pct_change()
    r_passo = (1 + sem[c].dropna().pct_change()) * passos[p][0] - 1
    conferir((r_aj - r_passo).abs().max() < 1e-8,
             f"{p}: retorno ajustado = retorno sem ajuste x degrau "
             "do fator, em todos os pregões")

print("\nProventos por ação implícitos no fator, por ano "
      "(leitura de sanidade).")
print("Soma, no ano, de (queda do fator na data ex) x (fechamento "
      "da véspera).\n(a) em reais por ação na base de hoje "
      "(corrigida por desdobramentos):")
reais, rend = {}, {}
for p in PAPEIS:
    c = p + ".SA"
    s = sem[c].dropna()
    queda = 1 - 1 / passos[p][0]
    queda = queda[queda > 1e-9]
    cisao = FATOR_CISAO if p == "ITUB4" else 1.0
    base = s.shift(1).reindex(queda.index)
    base = base.where(base.index >= DATA_CISAO, base / cisao)
    reais[p] = (queda * base).groupby(queda.index.year).sum()
    rend[p] = queda.groupby(queda.index.year).sum()
print(pd.DataFrame(reais).reindex(range(2010, 2026)).fillna(0)
      .round(4).to_string())
print("\n(b) como fração do preço da véspera (soma das quedas do "
      "fator no ano):")
print(pd.DataFrame(rend).reindex(range(2010, 2026)).fillna(0)
      .to_string(float_format="{:.2%}".format))

# ---------------------------------------------------------------
titulo("2. Amostra de datas: fechamento B3, valores novos, valores "
       "anteriores, retornos")


def linha_amostra(p, d):
    c, d = p + ".SA", pd.Timestamp(d)
    r = lambda s: ret(s).get(d, np.nan)
    return [p, d.date(), fech[p].get(d, np.nan), sem.at[d, c],
            aj.at[d, c], sem_ant[c].get(d, np.nan),
            aj_ant[c].get(d, np.nan), r(fech[p]), r(sem[c]),
            r(aj[c]), r(sem_ant[c])]


COLS = ["papel", "data", "fech_B3", "sem_novo", "aj_novo",
        "sem_ant", "aj_ant", "ret_B3", "ret_sem_novo",
        "ret_aj_novo", "ret_sem_ant"]
FMT = {c: "{:+.2%}".format for c in COLS[7:]}
FMT.update({c: "{:.4f}".format for c in COLS[3:7]})
FMT["fech_B3"] = "{:.2f}".format
print("ret_B3 é o retorno do fechamento bruto (inclui o degrau em "
      "data de evento);\nsem_ant e aj_ant são os valores dos "
      "arquivos anteriores (Yahoo limpo).\n")
print("2a. Datas citadas no livro")
cit = pd.DataFrame([linha_amostra(p, d) for p, ds in CITADAS.items()
                    for d in ds], columns=COLS)
print(cit.to_string(index=False, formatters=FMT))
print("\n2b. Primeiro pregão a partir do dia 2 de janeiro e de "
      "julho de cada ano")
amostra = []
for p in PAPEIS:
    dias = fech[p].dropna().index
    for ano in range(2010, 2026):
        for mes in (1, 7):
            d = dias[dias >= pd.Timestamp(ano, mes, 2)]
            if len(d) and d[0].year == ano and d[0].month == mes:
                amostra.append(linha_amostra(p, d[0]))
am = pd.DataFrame(amostra, columns=COLS)
print(am.to_string(index=False, formatters=FMT))
tudo = pd.concat([cit, am])
k = pd.Series([
    np.prod([e.f for e in eventos[(eventos["papel"] == p) & (
        eventos["data_ex"] > pd.Timestamp(d))].itertuples()])
    * (FATOR_CISAO if p == "ITUB4" and pd.Timestamp(d) < DATA_CISAO
       else 1.0) for p, d in zip(tudo["papel"], tudo["data"])])
erro = (tudo["sem_novo"].values / (tudo["fech_B3"].values * k)
        - 1).abs().max()
conferir(erro < 1e-8, f"nas {len(tudo)} linhas, sem_novo = fech_B3 "
         f"x fatores posteriores (erro máximo {erro:.1e})")

# ---------------------------------------------------------------
titulo("3. Retorno total (preço ajustado), por ano e de 2010 a "
       "2025: anterior x novo")


def anual(serie):
    s = serie.dropna()
    fim = s.groupby(s.index.year).last()
    ini = fim.shift(1)
    ini.iloc[0] = s.iloc[0]
    r = fim / ini - 1
    r["2010-25"] = s.iloc[-1] / s.iloc[0] - 1
    return r


grandes = []
for p in PAPEIS:
    c = p + ".SA"
    t = pd.DataFrame({"anterior": anual(aj_ant[c]),
                      "novo": anual(aj[c])})
    t["dif_pp"] = (t["novo"] - t["anterior"]) * 100
    t["nota"] = ""
    notas = []
    cor_p = correcoes[correcoes["papel"] == p]
    for ano in t.index[t["dif_pp"].abs() > 1]:
        motivo = EXPLICADAS.get((p, str(ano)))
        datas = [d.strftime("%d/%m/%Y") for d in cor_p["data_ex"]
                 if str(ano) in (str(d.year), "2010-25")]
        if datas and not motivo:
            motivo = ("correção de proventos da tabela (data ex "
                      + ", ".join(datas) + ")")
        t.loc[ano, "nota"] = "> 1 p.p."
        notas.append(f"  {ano}: {motivo or 'SEM EXPLICAÇÃO'}")
        grandes.append((p, ano, motivo))
    print(f"\n{p}")
    print(t.to_string(formatters={
        "anterior": "{:+.2%}".format, "novo": "{:+.2%}".format,
        "dif_pp": "{:+.2f}".format}))
    print("\n".join(notas))
print()
conferir(all(m for _, _, m in grandes),
         f"{len(grandes)} diferenças acima de 1 p.p., todas com "
         "explicação registrada")

# ---------------------------------------------------------------
titulo("4. Valores ausentes, pregões por ano e dias sem negócio")
acoes = [p + ".SA" for p in PAPEIS]
em_pregao = sem.index.isin(pregoes)
t = pd.DataFrame({
    "linhas": len(sem),
    "ausentes_novo": sem[acoes].isna().sum(),
    "fora_de_pregão": int((~em_pregao).sum()),
    "em_pregão_B3": sem.loc[em_pregao, acoes].isna().sum(),
    "ausentes_anterior": sem_ant[acoes].isna().sum(),
    "primeiro_negócio": [fech[p].first_valid_index().date()
                         for p in PAPEIS]})
print(t.to_string())
print("\nlinhas do arquivo:", len(sem), "| pregões da B3:",
      len(pregoes), "| dias sem pregão da B3 (só Yahoo):",
      int((~em_pregao).sum()))
conferir(sem.loc[~em_pregao, acoes + ["^BVSP"]].isna().all().all(),
         "fora de pregão da B3, ações e ^BVSP ausentes")
iguais = all(sem[p + ".SA"].reindex(pregoes).notna().equals(
    qtd[p].fillna(0) > 0) for p in PAPEIS)
conferir(iguais, "em pregão da B3 há preço se e somente se o "
         "COTAHIST registra quantidade negociada (QUATOT > 0)")
conferir(aj[acoes].isna().equals(sem[acoes].isna()),
         "ajustado e sem ajuste têm os mesmos ausentes")
print("\npregões da B3 por ano:")
print(pd.Series(pregoes.year).value_counts().sort_index()
      .to_frame("pregões").T.to_string())
print("\ndatas em que algum papel já listado não negociou:")
vazio = True
for p in PAPEIS:
    s = fech[p]
    falta = s[s.first_valid_index():].isna()
    if falta.any():
        vazio = False
        print(" ", p, [d.date().isoformat()
                       for d in falta[falta].index])
if vazio:
    print("  nenhuma: os oito papéis negociaram em todos os pregões "
          "desde a estreia\n  (MGLU3 estreou em 02/05/2011; antes de "
          "11/11/2013 ABEV3 é AMBV3).")
ib = sem.loc[em_pregao, "^BVSP"]
print("pregões da B3 sem ^BVSP no Yahoo:",
      [d.date().isoformat() for d in ib[ib.isna()].index])

# ---------------------------------------------------------------
titulo("5. Retornos diários que mudaram mais de 0,5 p.p. em relação "
       "aos arquivos anteriores")
print("Retorno entre cotações válidas consecutivas de cada arquivo "
      "(preço ajustado).\n'intervalo' = o retorno anterior cobria "
      "mais de um pregão, porque a limpeza\nantiga tinha apagado a "
      "véspera.\n")
mud = []
for p in PAPEIS:
    c = p + ".SA"
    for rotulo, n_, a_ in [("aj", aj, aj_ant), ("sem", sem, sem_ant)]:
        rn, ra = ret(n_[c]), ret(a_[c])
        va = a_[c].dropna().index
        vn = n_[c].dropna().index
        prev_a = pd.Series(va[:-1], index=va[1:])
        prev_n = pd.Series(vn[:-1], index=vn[1:])
        comum = rn.index.intersection(ra.index)
        d = pd.DataFrame({
            "papel": p, "base": rotulo, "novo": rn[comum],
            "anterior": ra[comum],
            "intervalo": prev_a[comum] != prev_n[comum]})
        d["dif_pp"] = (d["novo"] - d["anterior"]) * 100
        d["so_novo"] = len(rn.index.difference(ra.index))
        d["n"] = len(comum)
        mud.append(d)
mud = pd.concat(mud).rename_axis("data").reset_index()
for rotulo, nome in [("aj", "preço ajustado"),
                     ("sem", "preço sem ajuste")]:
    m = mud[mud["base"] == rotulo]
    g = m[m["dif_pp"].abs() > 0.5]
    print(f"{nome}: {len(m)} retornos comparáveis; {len(g)} mudaram "
          f"mais de 0,5 p.p. ({len(g) / max(len(m), 1):.2%}); "
          f"{int(g['intervalo'].sum())} deles por intervalo; "
          f"{int((m['dif_pp'].abs() > 1).sum())} acima de 1 p.p.; "
          f"{int((m['dif_pp'].abs() > 5).sum())} acima de 5 p.p.")
    novos = m.groupby("papel")["so_novo"].first()
    print(f"  retornos que só existem no arquivo novo (a data estava "
          f"ausente no anterior): {int(novos.sum())}")
m = mud[(mud["base"] == "aj") & (mud["dif_pp"].abs() > 0.5)]
print("\npor papel e ano (preço ajustado):")
t = pd.crosstab(m["data"].dt.year, m["papel"]).reindex(
    index=range(2010, 2026), columns=sorted(PAPEIS), fill_value=0)
t["total"] = t.sum(axis=1)
t.loc["total"] = t.sum()
print(t.to_string())
print("\nas vinte maiores mudanças (preço ajustado):")
top = m.reindex(m["dif_pp"].abs().sort_values(
    ascending=False).index).head(20)
print(top[["data", "papel", "anterior", "novo", "dif_pp",
           "intervalo"]].assign(data=top["data"].dt.date).to_string(
    index=False, formatters={"anterior": "{:+.2%}".format,
                             "novo": "{:+.2%}".format,
                             "dif_pp": "{:+.2f}".format}))

# ---------------------------------------------------------------
titulo("6. Volatilidade anualizada e autocorrelação de primeira "
       "ordem dos retornos diários")
print("Retornos simples do preço ajustado entre cotações válidas "
      "consecutivas.\n")
for ini, fim in [("2010", "2025"), ("2015", "2024"),
                 ("2011", "2015"), ("2016", "2025")]:
    t = {}
    for p in PAPEIS:
        c = p + ".SA"
        ra, rn = ret(aj_ant[c])[ini:fim], ret(aj[c])[ini:fim]
        t[p] = {"n_ant": len(ra), "n_novo": len(rn),
                "vol_ant": ra.std() * 252 ** 0.5,
                "vol_novo": rn.std() * 252 ** 0.5,
                "ac1_ant": ra.autocorr(1), "ac1_novo": rn.autocorr(1)}
    t = pd.DataFrame(t).T
    t["dif_vol_pp"] = (t["vol_novo"] - t["vol_ant"]) * 100
    t["dif_ac1"] = t["ac1_novo"] - t["ac1_ant"]
    print(f"{ini} a {fim}")
    print(t.to_string(formatters={
        "n_ant": "{:.0f}".format, "n_novo": "{:.0f}".format,
        "vol_ant": "{:.2%}".format, "vol_novo": "{:.2%}".format,
        "ac1_ant": "{:+.4f}".format, "ac1_novo": "{:+.4f}".format,
        "dif_vol_pp": "{:+.2f}".format,
        "dif_ac1": "{:+.4f}".format}), "\n")

# ---------------------------------------------------------------
titulo("7. Barras de PETR4 (petr4_ohlcv.csv)")
o = o_novo
bp = b3[b3["codigo"] == "PETR4"].set_index("data")
print("barras:", len(o), "| no arquivo anterior:", len(o_ant),
      "| pregões da B3:", len(pregoes))
conferir(o.index.equals(bp.index), "uma barra por pregão da B3 com "
         "negócio em PETR4")
conferir((o["Low"] <= o[["Open", "Close"]].min(axis=1)).all()
         and (o[["Open", "Close"]].max(axis=1) <= o["High"]).all(),
         "mínima <= abertura, fechamento <= máxima em todas as "
         "barras")
conferir((o["Volume"] > 0).all(), "nenhuma barra com volume zero "
         f"(mínimo: {o['Volume'].min():,} ações)")
conferir(np.allclose(o["Close"], aj["PETR4.SA"].reindex(o.index),
                     rtol=1e-9),
         "Close igual a PETR4.SA de precos_ajustados.csv")
k = o["Close"] / bp["fechamento"]
ok = all(np.allclose(o[c] / bp[v], k, rtol=1e-8) for c, v in [
    ("Open", "abertura"), ("High", "maxima"), ("Low", "minima")])
conferir(ok, "abertura, máxima e mínima oficiais com o mesmo fator "
         "do fechamento")
conferir((o["Volume"] == bp["quantidade"]).all(),
         "Volume = quantidade negociada oficial (PETR4 não teve "
         "desdobramento no período)")
ruins = ((o_ant["Close"] < o_ant["Low"] - 1e-9)
         | (o_ant["Close"] > o_ant["High"] + 1e-9)).sum()
print("  (arquivo anterior: barras com fechamento fora de "
      f"[mínima, máxima]: {int(ruins)})")

# ---------------------------------------------------------------
titulo("8. Reprodutibilidade: limpar_dados.py duas vezes, mesmo md5")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    shutil.copytree(NOVO / "bruto", tmp / "bruto")
    rodadas = []
    for _ in range(2):
        subprocess.run(
            [sys.executable, str(AQUI / "limpar_dados.py")],
            env={**os.environ, "DADOS_LIVRO": str(tmp)},
            check=True, capture_output=True)
        rodadas.append({g: md5(tmp / g) for g in GERADOS})
    for g in GERADOS:
        print(f"  {rodadas[0][g]}  {g}")
    conferir(rodadas[0] == rodadas[1],
             "as duas execuções geram os mesmos cinco arquivos")
    conferir(all(md5(NOVO / g) == rodadas[0][g] for g in GERADOS),
             "os arquivos entregues são os que o script gera a "
             "partir de bruto/")
print("\nmd5 dos arquivos brutos:")
for arq in sorted((NOVO / "bruto").glob("*.csv")):
    print(f"  {md5(arq)}  bruto/{arq.name}")

# ---------------------------------------------------------------
titulo("RESULTADO")
if falhas:
    print(f"{len(falhas)} conferência(s) falharam:")
    for f in falhas:
        print("  -", f)
    sys.exit(1)
print("todas as conferências obrigatórias passaram.")
