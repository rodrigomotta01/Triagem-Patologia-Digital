# 🧬 PathoShift Challenge — Guia do Time (Entrega 1)

## O que é o projeto?

Estamos construindo um **sistema que classifica imagens de tecido colorretal** (intestino grosso) automaticamente. O sistema recebe uma imagem microscópica e diz qual tipo de tecido está nela.

Isso é usado em **patologia digital** — em vez de um médico olhar cada lâmina no microscópio, o computador ajuda a classificar automaticamente.

> ⚠️ **Importante:** Não é ferramenta médica real. É um exercício acadêmico com dados públicos.

---

## Nosso Dataset

Usamos o **Kather Colorectal Histology** — 5.000 imagens (150×150 pixels) de tecido colorretal coradas com H&E (Hematoxilina & Eosina), divididas em **8 tipos de tecido**:

| Classe | O que é | Exemplo simples |
|--------|---------|-----------------|
| 🔴 TUMOR | Células cancerígenas | Células desorganizadas, núcleos grandes |
| 🟠 STROMA | Tecido de suporte | "Cimento" entre estruturas |
| 🟡 COMPLEX | Mistura complexa | Regiões com vários tipos misturados |
| 🟢 LYMPHO | Células de defesa | Bolinhas escuras concentradas |
| 🔵 DEBRIS | Restos celulares / muco | Material solto, sem estrutura |
| 🟣 MUCOSA | Mucosa saudável | Glândulas organizadas e regulares |
| ⚪ ADIPOSE | Gordura | Espaços vazios grandes e redondos |
| ⬜ EMPTY | Fundo / sem tecido | Área branca da lâmina |

São 625 imagens por classe (perfeitamente balanceado).

**Download:** [Kaggle — Colorectal Histology MNIST](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist) (~258 MB)

---

## Como funciona o pipeline? (Visão geral)

O sistema segue 6 etapas, como uma linha de montagem:

```
📁 Imagens → 🔍 Validação → 🎨 Pré-processamento → 📊 Extração de Features → 🤖 SVM → 📋 Resultado
```

Vamos detalhar cada uma:

### 1️⃣ Validação e Manifesto
**O que faz:** Abre todas as imagens, verifica se estão OK (não corrompidas, tamanho correto) e cria um "inventário" (manifesto JSONL).

**Por que:** Precisamos saber exatamente o que temos antes de treinar. O manifesto registra cada imagem com metadados para reprodutibilidade.

### 2️⃣ Pré-processamento
Dois procedimentos:

**a) Rejeição de fundo:** Detecta imagens que são só fundo branco (lâmina sem tecido). Converte para escala de cinza e conta quantos pixels são quase brancos. Se mais de 85% é branco → descarta.

**b) Normalização de cor (Reinhard):** Imagens de microscópio variam muito em cor dependendo do laboratório. A normalização pega uma imagem "referência" e ajusta todas as outras para terem cores parecidas. Funciona assim:
- Converte para o espaço L\*a\*b\* (separa luminosidade de cor)
- Calcula média e desvio de cor na referência
- Ajusta cada imagem para ter a mesma média e desvio

### 3️⃣ Extração de Features
Aqui é onde "traduzimos" cada imagem em **números que o classificador entende**. Extraímos 3 tipos:

| Feature | O que captura | Dimensão | Analogia simples |
|---------|---------------|----------|------------------|
| **Histograma HSV** | Distribuição de cores | 96 valores | "Quais cores aparecem e em que proporção?" |
| **LBP (textura)** | Padrões de textura local | 26 valores | "A superfície é lisa, granulada ou irregular?" |
| **Estatísticas globais** | Média de cor + complexidade | 7 valores | "No geral, é claro ou escuro? Simples ou complexo?" |

**Total: 129 números por imagem.** Esse vetor de 129 números é a "impressão digital" da imagem.

### 4️⃣ Classificação com SVM
O **SVM (Support Vector Machine)** recebe os 129 números de cada imagem e decide qual das 8 classes ela pertence.

Pense assim: imagine um gráfico com pontos de cores diferentes. O SVM encontra a melhor "linha" (na verdade um hiperplano em 129 dimensões) que separa os grupos. Usamos o kernel RBF, que permite fronteiras curvas.

Antes do SVM, um **StandardScaler** normaliza os 129 valores para que todos tenham a mesma escala (média=0, desvio=1). Sem isso, features com valores grandes dominariam.

### 5️⃣ Avaliação
Calculamos:
- **Acurácia:** % de acertos geral
- **Precision por classe:** dos que eu disse ser X, quantos realmente são X?
- **Recall por classe:** dos que são X, quantos eu encontrei?
- **F1-score:** equilíbrio entre precision e recall
- **F1-macro:** média simples do F1 de todas as classes (trata todas igualmente)
- **Matriz de confusão:** tabela mostrando acertos e erros entre classes

### 6️⃣ Análise de Erros
Identificamos as imagens que o modelo errou e analisamos padrões:
- Quais classes são mais confundidas?
- O modelo errou com alta confiança? (erros "convictos" são os piores)
- Os erros são em imagens de baixa qualidade?

---

## Estrutura do Projeto

```
PathoShift_Challenge/
├── src/pathoshift/          ← Todo o código do pipeline
│   ├── config.py            ← Configurações (hiperparâmetros, paths)
│   ├── cli.py               ← Comandos: prepare, predict, evaluate, describe
│   ├── data/                ← Leitura de dados e manifesto
│   ├── preprocessing/       ← Normalização de cor e controle de qualidade
│   ├── features/            ← Extração de features (HSV, LBP, stats)
│   ├── models/              ← SVM
│   ├── evaluation/          ← Métricas e análise de erros
│   └── utils/               ← Reprodutibilidade, timing
├── notebooks/01_eda.ipynb   ← Análise exploratória (gráficos e visualizações)
├── configs/default.yaml     ← Configuração padrão do experimento
├── outputs/                 ← Resultados (métricas, gráficos, modelo)
└── requirements.txt         ← Bibliotecas necessárias
```

---

## Conceitos-chave para a defesa técnica

Cada membro precisa saber explicar:

### 🎨 Espaços de Cor
- **RGB:** Vermelho, Verde, Azul — como o computador armazena cor
- **HSV:** Matiz, Saturação, Valor — separa "tipo de cor" de "brilho"
- **L\*a\*b\*:** Luminosidade, eixo verde-vermelho, eixo azul-amarelo — perceptualmente uniforme

### 📐 LBP (Local Binary Pattern)
- Para cada pixel, olha os vizinhos em círculo
- Compara: vizinho ≥ centro → 1, senão → 0
- Gera um código binário → histograma desses códigos = descrição de textura

### 🤖 SVM
- Encontra o hiperplano ótimo que separa as classes
- Kernel RBF permite fronteiras não-lineares
- Parâmetro C controla regularização (flexibilidade da fronteira)
- `class_weight='balanced'` compensa classes desbalanceadas

### 📊 Métricas
- **Acurácia sozinha não basta** — se 90% fosse uma classe, prever sempre ela daria 90%
- **F1-macro** é a métrica principal — média dos F1 de cada classe, trata todas igualmente
- **Matriz de confusão** mostra quais classes o modelo confunde

### 🔄 Reprodutibilidade
- **Seed fixa (42)** — garante que resultados são iguais entre execuções
- **Registro de versões** — salva versões do Python e bibliotecas
- **Config em YAML** — todos os hiperparâmetros documentados

---

## Tecnologias usadas

| Biblioteca | Para que usamos |
|-----------|-----------------|
| **scikit-learn** | SVM, métricas, split dos dados |
| **OpenCV** | Leitura de imagens, conversão de cores, histogramas |
| **scikit-image** | LBP (textura) |
| **matplotlib + seaborn** | Gráficos e visualizações |
| **NumPy** | Operações numéricas |
| **PyYAML** | Arquivo de configuração |

> 💡 **Não usamos GPU nem redes neurais nesta entrega.** Tudo roda na CPU com bibliotecas tradicionais de processamento de imagens.

---

## Como rodar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar o dataset do Kaggle e colocar em ./data/

# 3. Validar dataset e gerar manifesto
python -m pathoshift prepare --input ./data/kather --output ./outputs

# 4. Ver informações do sistema
python -m pathoshift describe
```

---

## Cronograma

| Fase | Dias | Entregáveis |
|------|------|-------------|
| Setup + Dados | 1–2 | Projeto criado, dataset baixado, manifesto gerado |
| Exploração | 3–4 | Notebook EDA com gráficos, pré-processamento pronto |
| Baseline SVM | 5–7 | Modelo treinado, avaliado, métricas geradas |
| Finalização | 8–9 | Análise de erros, CLI, documentação, revisão final |
