# Trabalho de Processamento de Imagens - PathoShift Challenge
## Triagem Robusta em Patologia Digital

**Disciplina:** Processamento de Imagens  
**Grupo:** 3 alunos  
**Entregas:** 3 etapas cumulativas (datas a definir)

---

## 1. Visão Geral

Cada grupo projetará, implementará e avaliará um pipeline de processamento e análise de imagens histopatológicas. O sistema receberá imagens microscópicas — inicialmente tiles (recortes) e, se a infraestrutura permitir, coleções organizadas por lâmina — e deverá produzir classificações, medidas de confiança e evidências visuais que permitam inspecionar a decisão.

Patologia digital é uma aplicação importante de processamento de imagens e visão computacional: a variabilidade de coloração, aquisição, qualidade e procedência das imagens torna insuficiente demonstrar desempenho apenas em uma partição aleatória de um único conjunto de dados. O objetivo é investigar, de modo reproduzível, como escolhas de pré-processamento, representação, classificação, agregação e validação afetam a robustez do sistema.

**O trabalho não tem finalidade de diagnóstico clínico nem autoriza uso clínico.**

---

## 2. Glossário

| Termo | Descrição |
|-------|-----------|
| Baseline | Implementação simples e reproduzível usada como ponto de comparação |
| Calibração | Correspondência entre confiança declarada e frequência de acerto |
| Domínio | Contexto de aquisição (laboratório, scanner, protocolo de coloração) |
| H&E | Coloração por hematoxilina e eosina |
| Lâmina digital / WSI | Whole Slide Image, potencialmente com gigapixels |
| Mudança de domínio | Alteração na distribuição entre treino e uso |
| Patch / tile | Recorte menor extraído de uma imagem ou lâmina |
| Split por paciente/lâmina | Separação que impede vazamento entre treino e teste |
| Validação externa | Avaliação em dados de fonte diferente do treinamento |

---

## 3. Objetivos de Aprendizagem

- Aplicar operações de processamento de imagens a um problema real, justificando as escolhas.
- Analisar cor, iluminação, textura, ruído, resolução, foco e artefatos de aquisição.
- Construir e comparar baselines de classificação de imagens.
- Organizar experimentos com separação apropriada entre desenvolvimento e avaliação.
- Medir qualidade preditiva, incerteza, custo computacional e robustez.
- Comunicar limitações, vieses e riscos de aplicação em saúde.

---

## 4. Entregas

| Entrega | Tema | Peso |
|---------|------|------|
| 1 | Pathology Image Pipeline | 25% |
| 2 | Slide-aware Analysis | 30% |
| 3 | Robust Pathology System | 45% |

---

## 5. Dados e Protocolo Experimental

- Datasets públicos, anonimizados, adequados ao escopo.
- Partições sem vazamento (separação por paciente/lâmina).
- Dados de teste privados não podem ser usados para hiperparâmetros.
- Classes podem ser desbalanceadas; acurácia isolada não é suficiente.
- Dataset da disciplina prevalece sobre versões de terceiros.

---

## 6. Interface de Execução

```bash
/pathoshift prepare --input /input --output /output
/pathoshift predict --input /input/manifest.jsonl --output /output/predictions.jsonl
/pathoshift evaluate --input /input/manifest.jsonl --output /output/metrics.json
/pathoshift describe
```

Sistema deve operar sem rede durante avaliação oficial.

---

## 7. Entrega 1 — Pathology Image Pipeline

**Requisitos:**
1. Leitura validada do conjunto e geração de manifesto reprodutível.
2. Análise exploratória breve: classes, dimensões, cor e exemplos de qualidade/artefatos.
3. Pelo menos um procedimento de pré-processamento justificado.
4. Pelo menos um baseline de classificação treinado e avaliado.
5. Métricas por classe e macro, matriz de confusão e exemplos de erros.
6. Registro de configuração, seed, versões e custo de execução.

**Componentes possíveis:** rejeição de fundo, avaliação de foco, RGB–HSV/LAB, normalização de intensidade/cor, filtragem, extração de textura, classificador clássico ou rede neural.

---

## 8. Entrega 2 — Slide-aware Analysis

- Estratégia de seleção, ordenação ou agregação de tiles.
- Predição no nível da tarefa: tile, região ou lâmina.
- Visualização conectando predição às regiões de evidência.
- Comparação com baseline sem a estratégia proposta.
- Análise de erros local e agregação global.

---

## 9. Entrega 3 — Robust Pathology System

Investigar questão de pesquisa formulada pelo grupo. Escolher ao menos um eixo:
- Mudança de domínio
- Normalização de cor vs. aumentação
- Baixa disponibilidade de rótulos
- Qualidade de imagem
- Calibração de confiança / detecção de incerteza
- Explicabilidade
- Eficiência computacional

---

## 10. Avaliação

- Correção, execução e reprodutibilidade
- Qualidade preditiva (métricas por classe e macro)
- Robustez em conjuntos não divulgados
- Qualidade das evidências visuais
- Custo computacional
- Arquitetura, documentação, análise de erros
- Defesa técnica individual

---

## 11. Uso de IA Generativa

- Ferramentas de IA podem ser usadas para programação, depuração, estudo.
- Responsabilidade integral do grupo.
- Declarar ferramentas usadas no relatório final.
- Sem chamadas a serviços externos durante avaliação.

---

## 12. Ética

- Não produz ferramenta médica.
- Explicitar vieses e limites de generalização.
- Dados não públicos ou identificáveis não podem ser usados.

