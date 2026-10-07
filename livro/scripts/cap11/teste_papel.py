"""Testes da corretora simulada: uma regra por teste."""
from types import SimpleNamespace as Barra

from papel import CorretoraPapel


def nova():
    return CorretoraPapel(10_000, comissao=0.001, slippage=0.01)


def barra(o, h, l, c, v=1000):
    return Barra(Open=o, High=h, Low=l, Close=c, Volume=v)


def perto(a, b):
    return abs(a - b) < 1e-9


def teste_mercado_no_open_com_slippage_e_comissao():
    c = nova()
    o = c.enviar("compra", 100)
    c.processar("d1", barra(10, 11, 9, 10.5))
    assert o.estado == "executada" and perto(o.preco_exec, 10.10)
    assert c.posicao == 100
    assert perto(c.caixa, 10_000 - 1010 - 1.01)


def teste_limitada_so_executa_se_atravessar():
    c = nova()
    o = c.enviar("compra", 100, "limitada", 9.50)
    c.processar("d1", barra(10, 10.2, 9.50, 10))     # só tocou
    assert o.estado == "aceita" and c.posicao == 0
    c.processar("d2", barra(10, 10.2, 9.49, 10))     # atravessou
    assert o.estado == "executada" and perto(o.preco_exec, 9.50)


def teste_limitada_com_abertura_no_limite_nao_executa():
    c = nova()
    o = c.enviar("compra", 100, "limitada", 9.50)
    c.processar("d1", barra(9.50, 9.80, 9.50, 9.70))  # só tocou
    assert o.estado == "aceita"


def teste_limitada_com_abertura_melhor_paga_slippage():
    c = nova()
    o = c.enviar("compra", 100, "limitada", 9.50)
    c.processar("d1", barra(9.20, 9.60, 9.10, 9.40))
    assert perto(o.preco_exec, 9.20 * 1.01)   # como a mercado
    o = c.enviar("compra", 100, "limitada", 9.25)
    c.processar("d2", barra(9.20, 9.60, 9.10, 9.40))
    assert perto(o.preco_exec, 9.25)          # limite é o teto


def teste_limitada_alta_nao_e_melhor_que_ordem_a_mercado():
    c = nova()
    lim = c.enviar("compra", 100, "limitada", 99.0)
    mer = c.enviar("compra", 100)
    c.processar("d1", barra(10, 11, 9, 10.5))
    assert perto(lim.preco_exec, mer.preco_exec)


def teste_limitada_de_venda():
    c = nova()
    c.enviar("compra", 200)
    c.processar("d1", barra(10, 10, 10, 10))
    o = c.enviar("venda", 100, "limitada", 11)
    c.processar("d2", barra(10, 11, 10, 10.5))       # só tocou
    assert o.estado == "aceita"
    c.processar("d3", barra(10.5, 11.2, 10.4, 11))
    assert perto(o.preco_exec, 11) and c.posicao == 100
    o = c.enviar("venda", 100, "limitada", 11)
    c.processar("d4", barra(12, 12.5, 11.8, 12))     # gap de alta
    assert perto(o.preco_exec, 12 * 0.99) and c.posicao == 0


def teste_stop_de_venda_no_gatilho_e_no_gap():
    c = nova()
    c.enviar("compra", 200)
    c.processar("d1", barra(10, 10, 10, 10))
    s1 = c.enviar("venda", 100, "stop", 9.00)
    c.processar("d2", barra(10, 10.1, 9.2, 9.5))     # não tocou
    assert s1.estado == "aceita"
    c.processar("d3", barra(9.5, 9.6, 8.9, 9.1))     # tocou
    assert perto(s1.preco_exec, 9.00 * 0.99)
    s2 = c.enviar("venda", 100, "stop", 9.00)
    c.processar("d4", barra(8.00, 8.2, 7.9, 8.1))  # gap
    assert perto(s2.preco_exec, 8.00 * 0.99)   # abaixo de 9


def teste_stop_de_compra():
    c = nova()
    o = c.enviar("compra", 100, "stop", 11)
    c.processar("d1", barra(10, 10.9, 9.9, 10.5))
    assert o.estado == "aceita"
    c.processar("d2", barra(10.5, 11, 10.4, 10.8))
    assert perto(o.preco_exec, 11 * 1.01)


def teste_rejeicoes_e_cancelamento():
    c = nova()
    assert c.enviar("compra", 0).estado == "rejeitada"
    assert c.enviar("compra", 10, "stop").estado == "rejeitada"
    caro = c.enviar("compra", 5000)                  # falta caixa
    vazio = c.enviar("venda", 100)                   # falta papel
    canc = c.enviar("compra", 100, "limitada", 5)
    c.cancelar(canc)
    c.processar("d1", barra(10, 11, 4, 10))
    assert caro.estado == vazio.estado == "rejeitada"
    assert canc.estado == "cancelada"
    assert c.posicao == 0 and c.caixa == 10_000 and not c.extrato


def teste_quantidade_e_preco_invalidos():
    c = nova()
    nan = float("nan")
    assert c.enviar("compra", 1.5).estado == "rejeitada"
    ruins = [c.enviar("compra", 1.5),
             c.enviar("compra", 10, "limitada", -1),
             c.enviar("compra", 10, "stop", nan)]
    assert all(o.estado == "rejeitada" for o in ruins)


def teste_barra_sem_volume_ou_sem_preco_nao_executa():
    c = nova()
    nan = float("nan")
    o = c.enviar("compra", 100)
    c.processar("d1", barra(10, 10, 10, 10, v=0))
    c.processar("d2", barra(nan, nan, nan, nan, v=nan))
    assert o.estado == "aceita" and c.caixa == 10_000


def teste_stop_e_alvo_na_mesma_barra_vale_o_stop():
    c = nova()
    c.enviar("compra", 100)
    c.processar("d1", barra(10, 10, 10, 10))
    stop = c.enviar("venda", 100, "stop", 9.50)   # vem antes
    alvo = c.enviar("venda", 100, "limitada", 10.80)
    c.processar("d2", barra(10, 10.90, 9.40, 10.20))
    assert perto(stop.preco_exec, 9.50 * 0.99)
    assert alvo.estado == "rejeitada" and c.posicao == 0


def teste_enviar_sozinho_nao_executa():
    c = nova()
    o = c.enviar("compra", 100)
    assert o.estado == "aceita" and c.posicao == 0
    assert not c.extrato


def teste_conservacao_do_patrimonio():
    c = nova()
    c.enviar("compra", 100)
    c.processar("d1", barra(10, 10, 10, 10))
    c.enviar("venda", 100)
    c.processar("d2", barra(10, 10, 10, 10))
    custo = 100 * 10 * (0.01 + 0.01) + 0.001 * (1010 + 990)
    assert perto(c.patrimonio(10), 10_000 - custo)


if __name__ == "__main__":
    testes = [f for n, f in sorted(globals().items())
              if n.startswith("teste_")]
    for f in testes:
        f()
    print(f"{len(testes)} testes passaram")
