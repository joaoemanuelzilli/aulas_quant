"""Corretora simulada (paper broker) sobre barras diárias OHLC."""
from dataclasses import dataclass
from math import isfinite


@dataclass
class Ordem:
    id: int
    lado: str                  # "compra" ou "venda"
    qtd: int
    tipo: str                  # "mercado", "limitada" ou "stop"
    preco: float = None        # limite ou gatilho
    estado: str = "aceita"
    preco_exec: float = None


class CorretoraPapel:
    def __init__(self, caixa, comissao=0.0003, slippage=0.0005):
        self.caixa, self.posicao = caixa, 0
        self.comissao, self.slippage = comissao, slippage
        self.ordens, self.extrato = [], []

    def enviar(self, lado, qtd, tipo="mercado", preco=None):
        o = Ordem(len(self.ordens) + 1, lado, qtd, tipo, preco)
        com_preco = tipo != "mercado"
        if (not isinstance(qtd, int) or qtd <= 0
                or lado not in ("compra", "venda")
                or tipo not in ("mercado", "limitada", "stop")
                or (com_preco and not (preco and isfinite(preco)
                                       and preco > 0))):
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
        """Preço de execução na barra b; None se não executa."""
        s = 1 if o.lado == "compra" else -1
        pior = 1 + s * self.slippage       # compra paga mais
        if o.tipo == "mercado":
            return b.Open * pior
        if o.tipo == "limitada":
            if s * (b.Open - o.preco) < 0:     # abriu melhor:
                p = b.Open * pior              # paga como mercado
                return s * min(s * p, s * o.preco)  # até o limite
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

    def processar(self, data, b):
        """Tenta executar na barra b as ordens já aceitas."""
        if not (b.Volume > 0 and b.Open > 0):  # barra inválida
            return
        for o in self.abertas():
            p = self._preco(o, b)
            if p is None:
                continue
            s = 1 if o.lado == "compra" else -1
            taxa = self.comissao * p * o.qtd
            sem_caixa = s == 1 and p * o.qtd + taxa > self.caixa
            sem_papel = s == -1 and o.qtd > self.posicao
            if sem_caixa or sem_papel:
                o.estado = "rejeitada"
                continue
            self.caixa -= s * p * o.qtd + taxa
            self.posicao += s * o.qtd
            o.estado, o.preco_exec = "executada", p
            self.extrato.append(
                (data, o.id, o.lado, o.tipo, o.qtd, p, taxa,
                 self.posicao, self.caixa))
