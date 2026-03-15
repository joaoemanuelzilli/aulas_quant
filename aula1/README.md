# Aula 1 — Núcleo Quant

Este repositório corresponde à **Aula 1 do Núcleo Quant**.

Nossa primeira aula será no **dia 14/03**, e os temas abordados são:

1. Apresentação da estrutura do núcleo.
2. Ambiente de trabalho (VS Code, Git).
3. Introdução a pandas e NumPy.
4. Importação e manipulação de séries de preços.
5. Cálculo de retornos simples e log-retornos.

Neste repositório da Aula 1 estão implementados os **temas 3, 4 e 5**.

---

## Onde cada tema aparece no projeto

### Tema 1 — Apresentação da estrutura do núcleo

Slides enviados no grupo LMF Quant.

### Tema 2 — Ambiente de trabalho (VS Code, Git)

Este tema aparece principalmente nas instruções de uso deste README, com foco em:

- abrir o projeto no VS Code;
- criar e ativar a `.venv`;
- instalar dependências pelo `requirements.txt`;
- selecionar o interpretador Python no VS Code.

Materiais de apoio:

- Vídeo ensinando baixar o VS Code, Python e configurar o GitHub:  
	https://www.youtube.com/watch?v=ypVQzgocNTU
- Link para instalar o VS Code:  
	https://code.visualstudio.com/download

### Tema 3 — Introdução a pandas e NumPy

Arquivos relacionados:

- `aula1/01_numpy_basico.py`
- `aula1/02_pandas_basico.py`
- `aula1/03_casos_reais.py`

### Tema 4 — Importação e manipulação de séries de preços

Arquivos relacionados:

- `aula1/series.py`
- `aula1/series2.py`

### Tema 5 — Cálculo de retornos simples e log-retornos

Arquivos relacionados:

- `aula1/logretorno.py`
- `aula1/plot.py`
- `aula1/retornosimples.py`

---

## Estrutura da aula

- `01_numpy_basico.py` → fundamentos de NumPy
- `02_pandas_basico.py` → fundamentos de pandas
- `03_casos_reais.py` → exemplos aplicados
- `series.py` e `series2.py` → importação e manipulação de séries de preços
- `retornosimples.py` e `logretorno.py` → cálculo de retornos
- `plot.py` → visualização gráfica dos retornos
- `dados/dados.csv` → base usada nos exemplos
- `outputs/` → arquivos gerados pelos scripts
- `requirements.txt` → dependências Python

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

### 4) Instalar dependências

```bash
pip install -r aula1/requirements.txt
```

### 5) Selecionar o interpretador no VS Code

1. Abra a paleta de comandos com `Ctrl+Shift+P`
2. Procure por `Python: Select Interpreter`
3. Escolha o interpretador da `.venv` na raiz do projeto

### 6) Executar os scripts

Navegue até a pasta da aula e rode os scripts:

```bash
cd aula1
python 01_numpy_basico.py
python 02_pandas_basico.py
python 03_casos_reais.py
python series.py
python series2.py
python retornosimples.py
python logretorno.py
python plot.py
```

---

## Solução de problemas comuns

### Erro de módulo não encontrado (`ModuleNotFoundError`)

- Confirme se a `.venv` está ativada
- Rode novamente `pip install -r aula1/requirements.txt` a partir da raiz do projeto

### Erro de arquivo não encontrado (`FileNotFoundError`)

- Execute os scripts de dentro da pasta `aula1` (`cd aula1`)
- Confirme se o arquivo `dados/dados.csv` existe

### VS Code não usa a `.venv`

- Refaça o passo `Python: Select Interpreter` apontando para a `.venv` na raiz do projeto
- Feche e abra o terminal integrado novamente

### O gráfico não abre na tela

Em alguns ambientes, o script pode salvar a imagem em vez de abrir uma janela interativa. Nesse caso, confira os arquivos gerados dentro de `outputs/`.
