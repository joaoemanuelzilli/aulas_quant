"""Corretora simulada (paper broker) sobre barras diárias."""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral


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


class CorretoraPapel:
    def __init__(self, caixa, comissao=0.0003, slippage=0.0005):
        self.caixa, self.posicao = caixa, 0
        self.comissao, self.slippage = comissao, slippage
        self.ordens, self.extrato = [], []

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
                preco and isfinite(preco) and preco > 0):
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

    def processar(self, data, b):
        """Tenta executar na barra b as ordens já aceitas."""
        if not (b.Volume > 0 and b.Open > 0):  # barra inválida
            return

        def vez(o):    # 1º a abertura; no dia, stops antes
            return (not self._na_abertura(o, b),
                    o.tipo != "stop", o.id)

        for o in sorted(self.abertas(), key=vez):
            p = self._preco(o, b)
            if p is None:
                continue
            s = 1 if o.lado == "compra" else -1
            taxa = self.comissao * p * o.qtd
            sem_caixa = s == 1 and p * o.qtd + taxa > self.caixa
            sem_papel = s == -1 and o.qtd > self.posicao
            if sem_caixa or sem_papel:
                o.estado = "rejeitada"
                o.motivo = "sem papel"
                if sem_caixa:
                    o.motivo = "sem caixa"
                continue
            self.caixa -= s * p * o.qtd + taxa
            self.posicao += s * o.qtd
            o.estado, o.preco_exec = "executada", p
            self.extrato.append(
                (data, o.id, o.lado, o.tipo, o.qtd, p, taxa,
                 self.posicao, self.caixa))
