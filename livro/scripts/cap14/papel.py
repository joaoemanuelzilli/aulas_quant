"""Corretora simulada (paper broker) sobre barras diárias."""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import NamedTuple


@dataclass
class Ordem:
    id: int
    lado: str                  # "compra" ou "venda"
    qtd: int
    tipo: str                  # "mercado", "limitada" ou "stop"
    preco: float = None        # limite ou gatilho
    estado: str = "aceita"
    motivo: str = ""           # por que foi rejeitada
    preco_exec: float = None


class Execucao(NamedTuple):
    """Uma linha do extrato; continua sendo uma tupla."""
    data: object
    ordem: int
    lado: str
    tipo: str
    qtd: int
    preco: float
    taxa: float
    posicao: int
    caixa: float


class CorretoraPapel:
    def __init__(self, caixa, comissao=0.0003, slippage=0.0005):
        self.caixa, self.posicao = caixa, 0
        self.comissao, self.slippage = comissao, slippage
        self.ordens, self.extrato = [], []
        self.data = None           # última barra processada

    def enviar(self, lado, qtd, tipo="mercado", preco=None):
        o = Ordem(len(self.ordens) + 1, lado, qtd, tipo, preco)
        inteiro = (isinstance(qtd, Integral)
                   and not isinstance(qtd, bool))
        if not (inteiro and qtd > 0):
            o.motivo = "quantidade inválida"
        elif (lado not in ("compra", "venda")
              or tipo not in ("mercado", "limitada", "stop")):
            o.motivo = "lado ou tipo desconhecido"
        elif tipo != "mercado" and not (
                isinstance(preco, Real)
                and not isinstance(preco, bool)
                and isfinite(preco) and preco > 0):
            o.motivo = "preço inválido"
        else:
            o.qtd = int(qtd)           # aceita inteiros do numpy
        if o.motivo:
            o.estado = "rejeitada"
        self.ordens.append(o)
        return o

    def cancelar(self, o):
        if o.estado == "aceita":
            o.estado = "cancelada"

    def abertas(self):
        return [o for o in self.ordens if o.estado == "aceita"]

    def patrimonio(self, preco):
        return self.caixa + self.posicao * preco

    def _preco(self, o, b):
        """Preço de execução na barra b, ou None."""
        s = 1 if o.lado == "compra" else -1
        pior = 1 + s * self.slippage       # compra paga mais
        if o.tipo == "mercado":
            return b.Open * pior
        if o.tipo == "limitada":
            if s * (b.Open - o.preco) < 0:     # abriu melhor:
                p = b.Open * pior              # paga como mercado
                return s * min(s * p, s * o.preco)  # teto: limite
            extremo = b.Low if s == 1 else b.High
            if s * (extremo - o.preco) < 0:    # atravessou
                return o.preco
            return None
        # stop: a compra dispara na alta; a venda, na queda
        if s * (b.Open - o.preco) >= 0:        # abriu além (gap)
            return b.Open * pior
        extremo = b.High if s == 1 else b.Low
        if s * (extremo - o.preco) >= 0:       # tocou o gatilho
            return o.preco * pior
        return None

    def _na_abertura(self, o, b):
        """A abertura da barra b já decide a execução de o?"""
        if o.tipo == "mercado":
            return True
        d = (1 if o.lado == "compra" else -1) * (b.Open - o.preco)
        return d < 0 if o.tipo == "limitada" else d >= 0

    @staticmethod
    def _valida(b):
        """Há negócio e os quatro preços são coerentes?"""
        precos = (b.Open, b.High, b.Low, b.Close)
        try:
            return bool(b.Volume > 0
                        and all(map(isfinite, precos))
                        and 0 < b.Low <= min(b.Open, b.Close)
                        and max(b.Open, b.Close) <= b.High)
        except TypeError:                  # None, texto etc.
            return False

    def processar(self, data, b):
        """Tenta executar na barra b as ordens já aceitas."""
        if not self._valida(b):            # barra inválida
            return
        # cada barra uma vez, em ordem: reprocessar uma barra
        # executaria ordens criadas depois de vê-la (look-ahead)
        assert self.data is None or data > self.data, data
        self.data = data

        def vez(o):    # 1º a abertura; no dia, stops antes
            return (not self._na_abertura(o, b),
                    o.tipo != "stop", o.id)

        fila = sorted(self.abertas(), key=vez)
        # stop de venda ainda sem papel: fica para o fim, porque
        # a compra que ele protege pode sair nesta mesma barra
        espera = [o for o in fila if o.tipo == "stop"
                  and o.lado == "venda" and o.qtd > self.posicao]
        resto = [o for o in fila if o not in espera]
        for o in resto + espera:
            self._executar(data, o, b)

    def _executar(self, data, o, b):
        p = self._preco(o, b)
        if p is None:
            return
        s = 1 if o.lado == "compra" else -1
        taxa = self.comissao * p * o.qtd
        # folga de 0,01 centavo: somas em float não são exatas
        custo = p * o.qtd + taxa
        sem_caixa = s == 1 and custo > self.caixa + 1e-4
        sem_papel = s == -1 and o.qtd > self.posicao
        if sem_caixa or sem_papel:
            o.estado = "rejeitada"
            o.motivo = "sem caixa" if sem_caixa else "sem papel"
            return
        self.caixa -= s * p * o.qtd + taxa
        self.posicao += s * o.qtd
        o.estado, o.preco_exec = "executada", p
        self.extrato.append(Execucao(
            data, o.id, o.lado, o.tipo, o.qtd, p, taxa,
            self.posicao, self.caixa))
