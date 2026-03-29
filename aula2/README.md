# Aula 2 — Núcleo Quant

Este repositório corresponde à **Aula 2 do Núcleo Quant**.

Nesta aula, os temas abordados foram:

1. yfinance e APIs.
2. Dados faltantes.
3. Visualização com Matplotlib/Plotly.
4. Análise exploratória de ativos.

Neste diretório da Aula 2, estão implementados esses temas de forma prática.

---

## Onde cada tema aparece no projeto

### Tema 1 — yfinance e APIs

Arquivos relacionados:

- `aula2/yfinance.py`
- `aula2/Matplotlib.py`
- `aula2/Plotly.py`
- `aula2/eda.py`

### Tema 2 — Dados faltantes

Arquivos relacionados:

- `aula2/interpolacao.py`
- `aula2/yfinance.py` (uso de `fillna`, `dropna`)
- `aula2/eda.py` (diagnostico e tratamento de NaN)

### Tema 3 — Visualizacao com Matplotlib/Plotly

Arquivos relacionados:

- `aula2/Matplotlib.py`
- `aula2/Plotly.py`
- `aula2/eda.py`

### Tema 4 — Analise exploratoria de ativos

Arquivos relacionados:

- `aula2/eda.py`
- `aula2/ex1_EDA.pdf`

---

## Estrutura da aula

- `yfinance.py` -> download de dados de mercado e retorno acumulado
- `interpolacao.py` -> exemplos de tratamento de dados faltantes
- `Matplotlib.py` -> graficos de preco, retorno e histograma com Matplotlib
- `Plotly.py` -> visualizacao interativa com Plotly
- `eda.py` -> EDA completa (retornos, volatilidade, correlacao, medias moveis e drawdown)
- `ex1_EDA.pdf` -> material complementar/exercicio da aula

---

## Passo a passo no VS Code

### 1) Abrir o projeto

No VS Code, abra a pasta raiz do projeto `Quant`.

### 2) Criar ambiente virtual

No terminal integrado, a partir da raiz do projeto (`Quant`):

#### Linux/macOS

```bash
python3 -m venv .venv
```

#### Windows (PowerShell)

```powershell
py -m venv .venv
```

### 3) Ativar ambiente virtual

#### Linux/macOS

```bash
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### 4) Instalar dependencias

```bash
pip install pandas numpy matplotlib plotly yfinance
```

### 5) Selecionar o interpretador no VS Code

1. Abra a paleta de comandos com `Ctrl+Shift+P`
2. Procure por `Python: Select Interpreter`
3. Escolha o interpretador da `.venv` na raiz do projeto

### 6) Executar os scripts

Navegue ate a pasta da aula e rode os scripts:

```bash
cd aula2
python interpolacao.py
python yfinance.py
python Matplotlib.py
python Plotly.py
python eda.py
```

---

## Solucao de problemas comuns

### Erro de modulo nao encontrado (`ModuleNotFoundError`)

- Confirme se a `.venv` esta ativada
- Rode novamente `pip install pandas numpy matplotlib plotly yfinance`

### Erro de rede/download no yfinance

- Verifique conexao com a internet
- Tente novamente apos alguns segundos (pode haver instabilidade temporaria na API)
- Confirme se os tickers usados nos scripts estao corretos

### Erro de arquivo nao encontrado (`FileNotFoundError`)

- Execute os scripts dentro de `aula2` (`cd aula2`)

### Grafico nao abre na tela

- Em alguns ambientes, a janela grafica pode nao abrir automaticamente
- Nesse caso, execute pelo VS Code com o interpretador da `.venv` selecionado
- Para Plotly, se necessario, rode no navegador com `fig.show()` ativo no script
