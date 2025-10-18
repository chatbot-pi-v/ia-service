# Arquitetura de Sistema RAG Multimodal para Conhecimento sobre Religiões Afro-Brasileiras

## Resumo

Este documento descreve a arquitetura de um sistema de Recuperação Aumentada por Geração (RAG - Retrieval-Augmented Generation) multimodal, desenvolvido para fornecer informações educacionais sobre religiões de matriz africana no Brasil. O sistema integra processamento de texto, áudio, vídeo e imagens, utilizando embeddings vetoriais e banco de dados vetorial para recuperação de informação contextual, aliado a modelos de linguagem de grande escala (LLMs) para geração de respostas.

---

## 1. Introdução

O sistema proposto implementa uma arquitetura RAG (Retrieval-Augmented Generation) com capacidades multimodais, permitindo o processamento e armazenamento de conteúdo em múltiplos formatos (texto, áudio, vídeo e imagens) e a geração de respostas contextualizadas através de um chatbot especializado. A solução foi desenvolvida para democratizar o acesso ao conhecimento sobre religiões afro-brasileiras, combatendo desinformação e preconceitos através de uma abordagem educacional e culturalmente sensível.

---

## 2. Arquitetura Geral do Sistema

### 2.1 Visão Geral

O sistema é composto por dois fluxos principais:

1. **Fluxo de Ingestão de Dados** (Offline): Processamento e armazenamento de conteúdo multimodal
2. **Fluxo de Inferência** (Online): Recuperação de informação e geração de respostas

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARQUITETURA GERAL DO SISTEMA                  │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │   DASHBOARD DE       │
                    │   ADMINISTRAÇÃO      │
                    └──────────┬───────────┘
                               │
                               │ Upload de Conteúdo
                               ▼
        ┌──────────────────────────────────────────┐
        │       MÓDULO DE INGESTÃO                 │
        │                                          │
        │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
        │  │  Áudio  │  │  Vídeo  │  │  Texto  │ │
        │  └────┬────┘  └────┬────┘  └────┬────┘ │
        │       │            │            │       │
        │       ▼            ▼            ▼       │
        │  ┌────────────────────────────────────┐ │
        │  │   PROCESSADORES ESPECIALIZADOS     │ │
        │  │   - Transcrição (Whisper)          │ │
        │  │   - Extração de frames             │ │
        │  │   - Segmentação textual            │ │
        │  └────────────────┬───────────────────┘ │
        └───────────────────┼─────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────┐
        │       MÓDULO DE EMBEDDINGS              │
        │                                         │
        │  ┌──────────────┐  ┌─────────────────┐ │
        │  │ Text Encoder │  │  Image Encoder  │ │
        │  │ BGE-base     │  │  CLIP ViT-B/32  │ │
        │  │ (768D)       │  │  (512D)         │ │
        │  └──────┬───────┘  └────────┬────────┘ │
        └─────────┼──────────────────────┼────────┘
                  │                      │
                  ▼                      ▼
        ┌─────────────────────────────────────────┐
        │     MILVUS VECTOR DATABASE              │
        │                                         │
        │  ┌─────────────────┐  ┌──────────────┐ │
        │  │ Text Collection │  │ Image        │ │
        │  │ IVF_FLAT Index  │  │ Collection   │ │
        │  │ ~15 PDFs        │  │ ~6 imagens   │ │
        │  └─────────────────┘  └──────────────┘ │
        └─────────────────────────────────────────┘
                            │
                            │
                    ┌───────▼───────┐
                    │  CHATBOT API  │
                    │  (Flask)      │
                    └───────┬───────┘
                            │
                            ▼
        ┌─────────────────────────────────────────┐
        │      PIPELINE RAG                       │
        │                                         │
        │  1. Validação de Relevância             │
        │  2. Recuperação de Contexto (k=3)       │
        │  3. Busca Multimodal de Imagens         │
        │  4. Geração via LLM (Llama3-8b)         │
        └─────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   RESPOSTA    │
                    │ Texto + Imagem│
                    └───────────────┘
```

### 2.2 Componentes Tecnológicos

| Componente | Tecnologia | Função |
|------------|-----------|---------|
| API Backend | Flask | Orquestração de requisições |
| Vector Database | Milvus | Armazenamento e busca vetorial |
| Text Embeddings | BAAI/bge-base-en-v1.5 | Vetorização de texto (768D) |
| Image Embeddings | CLIP ViT-B/32 | Vetorização de imagens (512D) |
| LLM | Groq Llama3-8b-8192 | Geração de respostas |
| Processamento PDF | LangChain + PyPDF | Extração e segmentação textual |
| Transcrição Áudio | Whisper (inferido) | Conversão áudio → texto |

---

## 3. Módulo de Ingestão de Dados

### 3.1 Fluxo de Alimentação via Dashboard

O sistema recebe conteúdo multimodal através de um dashboard administrativo, permitindo o upload de:

- **Documentos textuais** (PDF, TXT, DOCX)
- **Arquivos de áudio** (entrevistas, palestras, depoimentos)
- **Arquivos de vídeo** (documentários, rituais, aulas)
- **Imagens** (fotografias de rituais, objetos sagrados, terreiros)

```
┌─────────────────────────────────────────────────────────────────┐
│            FLUXO DE INGESTÃO MULTIMODAL                          │
└─────────────────────────────────────────────────────────────────┘

DASHBOARD
    │
    ├─→ [ÁUDIO] (.mp3, .wav, .m4a)
    │       │
    │       ├─→ Transcrição (Whisper API)
    │       │       │
    │       │       └─→ Texto transcrito
    │       │               │
    │       │               └─→ Segmentação em chunks
    │       │
    │       └─→ Metadados (duração, speaker, data)
    │
    ├─→ [VÍDEO] (.mp4, .avi, .mov)
    │       │
    │       ├─→ Extração de áudio → Transcrição
    │       │       │
    │       │       └─→ Texto transcrito → Chunks
    │       │
    │       ├─→ Extração de frames-chave
    │       │       │
    │       │       └─→ Imagens → CLIP embeddings
    │       │
    │       └─→ Metadados (duração, resolução, data)
    │
    ├─→ [TEXTO] (.pdf, .txt, .docx)
    │       │
    │       ├─→ Extração de texto (PyPDF/python-docx)
    │       │       │
    │       │       └─→ Segmentação em chunks
    │       │
    │       └─→ Metadados (autor, título, páginas)
    │
    └─→ [IMAGENS] (.jpg, .png, .jpeg)
            │
            ├─→ Geração de caption (CLIP/BLIP)
            │       │
            │       └─→ Armazenamento do caption
            │
            ├─→ Extração de embedding CLIP (512D)
            │
            └─→ Metadados (dimensões, data, origem)
                    │
                    ▼
            ┌───────────────────┐
            │ BANCO DE VETORES  │
            │     (Milvus)      │
            └───────────────────┘
```

### 3.2 Processamento Específico por Modalidade

#### 3.2.1 Processamento de Documentos Textuais (PDF)

**Arquivo:** `processing/pdf_loader.py`

```python
# Configuração de chunking
Chunk Size: 700 caracteres
Overlap: 150 caracteres
Método: RecursiveCharacterTextSplitter
```

**Justificativa técnica:**
- **Chunk size de 700 caracteres**: Balanceia granularidade semântica e limite de contexto do modelo de embeddings
- **Overlap de 150 caracteres** (~21%): Garante continuidade contextual entre chunks adjacentes, preservando informações que podem ser divididas nas fronteiras
- **Método recursivo**: Prioriza quebras naturais (parágrafos, sentenças) antes de forçar divisão por caracteres

**Pipeline de processamento:**

1. **Leitura**: PyPDFLoader extrai texto página por página
2. **Segmentação**: RecursiveCharacterTextSplitter divide em chunks
3. **Limpeza**: Remoção de caracteres especiais e normalização
4. **Metadados**: Preserva informações de origem (arquivo, página)
5. **Embedding**: Vetorização via BAAI/bge-base-en-v1.5

#### 3.2.2 Processamento de Áudio

**Fluxo inferido para expansão:**

```
Áudio (.mp3/.wav/.m4a)
    ↓
[1] Transcrição via Whisper
    • Modelo: whisper-large-v3
    • Idioma: pt-BR
    • Timestamp: Sim
    ↓
[2] Texto transcrito
    ↓
[3] Segmentação temporal
    • Dividir por pausas/sentenças
    • Manter timestamp de referência
    ↓
[4] Chunking textual
    • Mesmo processo dos PDFs
    • Chunk size: 700 caracteres
    • Overlap: 150 caracteres
    ↓
[5] Embedding (BGE 768D)
    ↓
[6] Armazenamento no Milvus
    • Metadados: arquivo_origem, timestamp_inicio, timestamp_fim
```

#### 3.2.3 Processamento de Vídeo

**Fluxo inferido para expansão:**

```
Vídeo (.mp4/.avi/.mov)
    ↓
[1] Separação de modalidades
    ├─→ Áudio → Transcrição (processo 3.2.2)
    │
    └─→ Visual
        ↓
[2] Extração de frames-chave
    • Algoritmo: Diferença de histograma
    • Taxa: ~1 frame a cada 10-30 segundos
    • Ou: Detecção de mudança de cena
    ↓
[3] Processamento de frames
    ├─→ Geração de caption (BLIP-2/CLIP)
    └─→ Embedding CLIP (512D)
    ↓
[4] Armazenamento
    • Frames → Coleção de imagens
    • Transcrição → Coleção de texto
    • Linking: video_id comum
```

#### 3.2.4 Processamento de Imagens

**Arquivo:** `milvus_img.py`

**Implementação atual:**

```python
Modelo: CLIP ViT-B/32
Dimensão: 512
Normalização: L2
Métrica: Inner Product (IP)
```

**Pipeline:**

```
Imagem (.jpg/.png)
    ↓
[1] Pré-processamento
    • Redimensionamento: 224x224
    • Normalização de pixels
    • Conversão RGB
    ↓
[2] Extração de embedding
    • Encoder: CLIP ViT-B/32
    • Saída: vetor 512D
    • Normalização L2
    ↓
[3] Caption (manual ou automático)
    • Manual: via dashboard
    • Automático: CLIP/BLIP-2
    ↓
[4] Armazenamento no Milvus
    • Fields: id, image_path, vector, captions
```

---

## 4. Arquitetura de Embeddings

### 4.1 Embeddings Textuais

**Modelo:** BAAI/bge-base-en-v1.5 (Beijing Academy of Artificial Intelligence)

**Especificações técnicas:**

| Parâmetro | Valor |
|-----------|-------|
| Dimensionalidade | 768 |
| Modelo base | BERT |
| Vocabulário | 30.522 tokens |
| Camadas | 12 transformer layers |
| Cabeças de atenção | 12 |
| Parâmetros | ~110M |
| Normalização | L2 |
| Device | CPU |

**Justificativa da escolha:**
- **Alta performance** em tarefas de recuperação semântica (top-3 no MTEB Leaderboard)
- **Eficiência computacional** para deployment em CPU
- **Suporte multilíngue** (embora otimizado para inglês, funciona razoavelmente com português)
- **Open-source** e licença permissiva

**Processo de embedding:**

```python
# Pseudocódigo
texto = "Orixá é uma divindade das religiões afro-brasileiras..."

# 1. Tokenização
tokens = tokenizer.encode(texto)
# Output: [101, 2023, 2003, 2019, 8523, 102, ...]

# 2. Embedding via transformer
hidden_states = model(tokens)
# Shape: (batch_size, sequence_length, 768)

# 3. Pooling (mean pooling)
embedding = mean_pool(hidden_states)
# Shape: (768,)

# 4. Normalização L2
embedding = embedding / ||embedding||_2
# Magnitude: 1.0
```

### 4.2 Embeddings Visuais

**Modelo:** CLIP ViT-B/32 (OpenAI)

**Especificações técnicas:**

| Parâmetro | Valor |
|-----------|-------|
| Dimensionalidade | 512 |
| Arquitetura | Vision Transformer |
| Patch size | 32x32 pixels |
| Resolução input | 224x224 |
| Camadas | 12 |
| Parâmetros (visual) | ~86M |
| Treinamento | 400M pares imagem-texto |
| Normalização | L2 |

**Justificativa da escolha:**
- **Multimodalidade nativa**: Espaço latente compartilhado entre texto e imagem
- **Zero-shot capabilities**: Generaliza bem para domínios específicos
- **Robustez**: Treinado em dataset massivo e diversificado
- **Busca texto-imagem**: Permite query textual em coleção de imagens

**Processo de embedding (imagem):**

```python
# Pseudocódigo
imagem = Image.open("atabaque.jpg")

# 1. Pré-processamento
imagem_processada = preprocess(imagem)
# - Resize para 224x224
# - Normalização: mean=[0.48145466, 0.4578275, 0.40821073]
#                std=[0.26862954, 0.26130258, 0.27577711]

# 2. Divisão em patches
patches = dividir_em_patches(imagem_processada, patch_size=32)
# Output: 49 patches (7x7 grid)

# 3. Embedding via Vision Transformer
embedding = vision_encoder(patches)
# Shape: (512,)

# 4. Normalização L2
embedding = embedding / ||embedding||_2
```

**Processo de embedding (texto → imagem):**

```python
# Para busca texto-imagem
query = "atabaque instrumento musical sagrado"

# 1. Tokenização
tokens = clip.tokenize([query])

# 2. Embedding via text encoder
text_embedding = text_encoder(tokens)
# Shape: (512,) - MESMO ESPAÇO LATENTE da imagem!

# 3. Normalização L2
text_embedding = text_embedding / ||text_embedding||_2

# 4. Busca por similaridade
# Cosine similarity = dot product (vetores normalizados)
```

---

## 5. Banco de Dados Vetorial (Milvus)

### 5.1 Arquitetura do Milvus

O Milvus é um banco de dados vetorial distribuído, otimizado para busca de similaridade em alta dimensionalidade.

**Configuração do sistema:**

```yaml
Versão: Milvus 2.x
Deployment: Cloud (Zilliz Cloud inferido)
Persistência: Sim
Consistency Level: Strong
```

### 5.2 Coleção de Texto

**Schema:**

```python
Collection Name: {COLLECTION_NAME}  # Configurável via env

Fields:
  - id: INT64 (Primary Key, Auto-ID)
  - vector: FLOAT_VECTOR (dim=768)
  - metadata: JSON (opcional - contém source, page, etc.)

Index:
  Type: IVF_FLAT
  Metric: IP (Inner Product)
  Parameters:
    nlist: 128  # Número de clusters
```

**Justificativa do índice IVF_FLAT:**
- **IVF (Inverted File)**: Particiona o espaço vetorial em clusters (Voronoi cells)
- **FLAT**: Busca exaustiva dentro de cada cluster (maior precisão)
- **Trade-off**: Balanceia velocidade e recall para datasets médios (~10K-1M vetores)
- **Inner Product (IP)**: Equivalente a cosine similarity para vetores normalizados (mais eficiente computacionalmente)

**Estatísticas da coleção:**

```
Documentos fonte: 15 PDFs
Chunks estimados: ~1.500-2.000 (depende do tamanho dos PDFs)
Tamanho médio do chunk: 700 caracteres
Armazenamento: ~12MB (apenas vetores)
```

### 5.3 Coleção de Imagens

**Schema:**

```python
Collection Name: image_embeddings

Fields:
  - id: INT64 (Primary Key, Auto-ID)
  - image_path: VARCHAR(512)
  - vector: FLOAT_VECTOR (dim=512)
  - captions: VARCHAR(1024)

Index:
  Type: FLAT (ou IVF_FLAT)
  Metric: IP
```

**Justificativa:**
- Dataset pequeno (~6 imagens atualmente) → FLAT index suficiente
- Escalável para milhares de imagens com IVF_FLAT

**Estatísticas da coleção:**

```
Imagens: 6
Armazenamento: ~12KB (vetores) + ~6MB (imagens originais)
Captions: Manuais e descritivos
```

### 5.4 Operações de Busca

**Busca por similaridade textual:**

```python
# Configuração
top_k = 3
metric_type = "IP"  # Inner Product
search_params = {"nprobe": 10}  # Clusters a pesquisar

# Query
query_vector = embedding_model.encode(user_question)
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param=search_params,
    limit=top_k,
    output_fields=["metadata"]
)

# Output
# [
#   (doc_1, score=0.89),
#   (doc_2, score=0.76),
#   (doc_3, score=0.65)
# ]
```

**Busca multimodal (texto → imagem):**

```python
# Query textual
query_text = "atabaque instrumento musical"

# Encode com CLIP text encoder
text_vector = clip_model.encode_text(query_text)

# Busca na coleção de imagens
results = image_collection.search(
    data=[text_vector],
    anns_field="vector",
    limit=3,
    output_fields=["image_path", "captions"]
)

# Filtragem adicional
filtered = [r for r in results if r.distance < 0.3]
```

---

## 6. Pipeline RAG (Retrieval-Augmented Generation)

### 6.1 Arquitetura do Pipeline

**Arquivo:** `services/rag_pipeline.py` - Classe `SafeRAGPipeline`

O pipeline RAG implementa uma abordagem em 4 etapas com safeguards para garantir qualidade e relevância.

```
┌─────────────────────────────────────────────────────────────┐
│                  PIPELINE RAG DETALHADO                      │
└─────────────────────────────────────────────────────────────┘

USER QUERY: "O que é um Orixá?"
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ ETAPA 1: VALIDAÇÃO DE RELEVÂNCIA                     │
│ ──────────────────────────────────────────────────   │
│                                                      │
│  Objetivo: Filtrar perguntas fora do domínio        │
│                                                      │
│  [1.1] Encode query com SentenceTransformer         │
│        Model: BAAI/bge-base-en-v1.5                 │
│        Output: query_embedding (768D)                │
│                                                      │
│  [1.2] Comparar com embeddings de exemplos          │
│        Exemplos no domínio:                          │
│        - "O que é um Orixá?"                        │
│        - "Como funciona uma gira na Umbanda?"       │
│        - "Qual a origem do Candomblé?"              │
│        ... (10 exemplos)                             │
│                                                      │
│        domain_embeddings = encode(exemplos)          │
│        similarities = cosine_sim(query, domain)      │
│        max_score = max(similarities)                 │
│                                                      │
│  [1.3] Decisão:                                     │
│        IF max_score >= 0.74:                         │
│            → CONTINUE (relevante)                    │
│        ELSE:                                         │
│            → RETURN mensagem educada de rejeição    │
│                                                      │
│  Resultado: ✓ PASS (score = 0.92)                  │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ ETAPA 2: RECUPERAÇÃO DE CONTEXTO TEXTUAL             │
│ ──────────────────────────────────────────────────   │
│                                                      │
│  [2.1] Busca por similaridade no Milvus             │
│        query_embedding = encode(query)               │
│        results = milvus.search(                      │
│            vector=query_embedding,                   │
│            top_k=3,                                  │
│            metric="IP"                               │
│        )                                             │
│                                                      │
│  [2.2] Resultados brutos:                           │
│        Doc 1: "Orixá é uma divindade..." | score: 0.15 │
│        Doc 2: "Na cosmologia iorubá..." | score: 0.42  │
│        Doc 3: "Cada Orixá possui..."    | score: 0.78  │
│                                                      │
│  [2.3] Ponderação adaptativa:                       │
│        weight = max(0.1, 1 - (score / max_score))   │
│                                                      │
│        Doc 1: weight = 0.85 → alta relevância       │
│        Doc 2: weight = 0.58 → média relevância      │
│        Doc 3: weight = 0.22 → baixa relevância      │
│                                                      │
│  [2.4] Cropping de contexto:                        │
│        • Divide texto em sentenças                   │
│        • Mantém N sentenças: N = ceil(total * weight)│
│        • Max 500 caracteres por chunk                │
│                                                      │
│        Doc 1: 85% do texto (alta relevância)         │
│        Doc 2: 58% do texto                           │
│        Doc 3: 22% do texto (apenas início)           │
│                                                      │
│  [2.5] Concatenação:                                │
│        context = "\n\n".join([doc1, doc2, doc3])    │
│                                                      │
│  Resultado: context_text (~1200 caracteres)         │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ ETAPA 3: RECUPERAÇÃO MULTIMODAL (IMAGEM)             │
│ ──────────────────────────────────────────────────   │
│                                                      │
│  [3.1] Encode query com CLIP text encoder           │
│        text_embedding = clip.encode_text(query)      │
│        Output: text_vector (512D, L2 normalized)     │
│                                                      │
│  [3.2] Busca na coleção de imagens                  │
│        results = milvus_img.search(                  │
│            vector=text_embedding,                    │
│            top_k=3,                                  │
│            output_fields=["image_path", "captions"]  │
│        )                                             │
│                                                      │
│  [3.3] Resultados:                                  │
│        Img 1: "trio de atabaques" | distance: 0.24   │
│        Img 2: "terreiro.jpeg"     | distance: 0.67   │
│        Img 3: "buzios.jpg"        | distance: 0.82   │
│                                                      │
│  [3.4] Filtragem inteligente:                       │
│        • Threshold: distance < 0.3                   │
│        • Keyword matching:                           │
│          query_keywords = {"orixá", "divindade"}     │
│          caption_keywords = {"atabaque", "trio"}     │
│          overlap = query_keywords ∩ caption_keywords │
│                                                      │
│        IF distance < 0.3 AND overlap > 0:            │
│            → ACEITAR imagem                          │
│                                                      │
│  [3.5] Seleção final:                               │
│        selected_image = Img 1 (passa nos filtros)    │
│                                                      │
│  [3.6] Conversão para base64:                       │
│        image_path = "docs/images/atabaque.jpg"       │
│        image_base64 = base64.encode(image_bytes)     │
│                                                      │
│  Resultado: image_base64 + caption                  │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ ETAPA 4: GERAÇÃO VIA LLM                             │
│ ──────────────────────────────────────────────────   │
│                                                      │
│  [4.1] Montagem do contexto final:                  │
│        final_context = context_text +                │
│                       "\n\nLegenda: " + caption      │
│                                                      │
│  [4.2] Construção do prompt:                        │
│        ┌────────────────────────────────────┐       │
│        │ # SISTEMA                           │       │
│        │ Você é Nanã, uma guia sábia...     │       │
│        │ [Persona + Diretrizes]              │       │
│        │                                     │       │
│        │ # CONTEXTO                          │       │
│        │ {final_context}                     │       │
│        │                                     │       │
│        │ # PERGUNTA                          │       │
│        │ {user_question}                     │       │
│        └────────────────────────────────────┘       │
│                                                      │
│  [4.3] Configuração do LLM:                         │
│        Provider: Groq                                │
│        Model: llama3-8b-8192                         │
│        Temperature: 0 (determinístico)               │
│        Max tokens: 8192                              │
│        Timeout: 10s                                  │
│                                                      │
│  [4.4] Invocação:                                   │
│        response = llm.invoke(prompt)                 │
│                                                      │
│  [4.5] Parsing:                                     │
│        answer_text = StrOutputParser()(response)     │
│                                                      │
│  Resultado: answer_text (~300-500 palavras)         │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ COMPOSIÇÃO DA RESPOSTA FINAL                         │
│ ──────────────────────────────────────────────────   │
│                                                      │
│  {                                                   │
│    "answer": "Orixá é uma divindade das religiões...",│
│    "image_base64": "iVBORw0KGgo...",                │
│    "image_caption": "trio de atabaques"             │
│  }                                                   │
└──────────────────────────────────────────────────────┘
```

### 6.2 Algoritmo de Ponderação Adaptativa

Um dos diferenciais do sistema é o **cropping adaptativo de contexto** baseado em relevância:

```python
def score_to_weight(score, max_score=1.0):
    """
    Converte score de distância em peso de relevância
    
    Scores menores = maior similaridade = maior peso
    Score 0.1 → weight 0.9
    Score 0.9 → weight 0.1
    """
    weight = max(0.1, 1 - (score / max_score))
    return weight

def crop_context(text, weight, max_len=500):
    """
    Corta o contexto proporcionalmente à relevância
    
    Documentos mais relevantes: mantém mais conteúdo
    Documentos menos relevantes: apenas início
    """
    sentences = text.split(". ")
    num_sentences = max(1, ceil(len(sentences) * weight))
    cropped = ". ".join(sentences[:num_sentences])
    return cropped[:max_len]
```

**Justificativa:**
- **Otimização de tokens**: Evita desperdício de contexto com informações marginalmente relevantes
- **Foco na qualidade**: Prioriza conteúdo altamente relevante
- **Prevenção de ruído**: Reduz chance de o LLM se confundir com informações periféricas

### 6.3 Modelo de Linguagem (LLM)

**Configuração:**

```yaml
Provider: Groq Cloud
Model: llama3-8b-8192
Context Window: 8.192 tokens
Temperature: 0
Top-p: 1.0
Timeout: 10 segundos
Request Timeout: 10 segundos
```

**Justificativa da escolha - Llama3-8b:**
- **Velocidade**: Inferência otimizada pela infraestrutura Groq (LPU)
- **Qualidade**: Performance comparável a modelos maiores em tarefas específicas
- **Contexto**: 8K tokens suficientes para o contexto recuperado + prompt
- **Custo**: Modelo open-source com API gratuita (tier Groq)
- **Português**: Bom suporte ao idioma (treinado em dataset multilíngue)

**Persona "Nanã":**

O sistema implementa uma persona culturalmente sensível e educacional:

```
Características:
- Inspirada em Nanã Buruquê (divindade da sabedoria)
- Mulher negra sábia e acolhedora
- Tom educativo, respeitoso e didático
- Combate preconceitos e desinformação
- Linguagem simples e acessível
- Uso moderado de expressões carinhosas

Diretrizes:
✓ Valorizar a pergunta do usuário
✓ Usar exemplos práticos e histórias
✓ Explicar termos técnicos
✓ Apresentar diferentes perspectivas (nações)
✓ Conectar com valores universais
✓ Corrigir termos pejorativos educadamente
✗ Não se apresentar explicitamente
✗ Evitar jargões sem explicação
```

---

## 7. API e Interface

### 7.1 Endpoint Principal

**Arquivo:** `api/routes.py`

```http
POST /question
Content-Type: application/json

Request:
{
  "question": "O que são os Orixás?"
}

Response:
{
  "answer": "Orixás são divindades das religiões de matriz africana...",
  "image_base64": "iVBORw0KGgoAAAANSU...",
  "image_caption": "trio de atabaques"
}
```

**Códigos de status:**

```
200 OK - Resposta gerada com sucesso
400 Bad Request - Pergunta vazia ou inválida
500 Internal Server Error - Erro no processamento
```

### 7.2 Fluxo de Requisição

```
Cliente → Flask App → RAG Service → RAG Pipeline
                            ↓
                    [Relevance Check]
                            ↓
                  ┌─────────┴─────────┐
                  ▼                   ▼
        [Text Retrieval]    [Image Retrieval]
           (Milvus)             (Milvus)
                  │                   │
                  └─────────┬─────────┘
                            ▼
                   [Context Assembly]
                            ↓
                      [LLM Generation]
                            ↓
                     [Response JSON]
                            ↓
                         Cliente
```

---

## 8. Métricas e Parâmetros de Configuração

### 8.1 Parâmetros de Chunking

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Chunk Size | 700 chars | Balanceia granularidade e contexto |
| Overlap | 150 chars | 21% de sobreposição para continuidade |
| Splitter | Recursive | Respeita estrutura natural do texto |

### 8.2 Parâmetros de Embeddings

| Componente | Dimensão | Modelo | Normalização |
|------------|----------|--------|--------------|
| Texto | 768 | BGE-base | L2 |
| Imagem | 512 | CLIP ViT-B/32 | L2 |

### 8.3 Parâmetros de Recuperação

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Top-K (texto) | 3 | Documentos recuperados |
| Top-K (imagem) | 3 | Imagens candidatas |
| Threshold (relevância) | 0.74 | Mínimo para pergunta válida |
| Threshold (imagem) | 0.3 | Distância máxima para imagem |
| Max context | 500 chars | Por chunk após cropping |
| Context total | ~1500 chars | Soma dos 3 chunks |

### 8.4 Parâmetros do LLM

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Temperature | 0 | Respostas determinísticas |
| Max tokens | 8192 | Contexto suficiente |
| Timeout | 10s | Limite de latência |
| Top-p | 1.0 | Sem sampling |

---

## 9. Fluxo de Dados Completo (End-to-End)

### 9.1 Fase de Ingestão (Offline)

```
[DASHBOARD] → Upload de conteúdo
       ↓
┌─────────────────────────────────────────┐
│ PROCESSAMENTO MULTIMODAL                │
├─────────────────────────────────────────┤
│ Texto → Chunking → Embedding (768D)    │
│ Áudio → Transcrição → Chunking → Emb   │
│ Vídeo → Áudio + Frames → Processamento │
│ Imagem → Caption → Embedding (512D)     │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ ARMAZENAMENTO VETORIAL                  │
├─────────────────────────────────────────┤
│ Milvus - Text Collection (768D)        │
│ Milvus - Image Collection (512D)       │
└─────────────────────────────────────────┘
```

### 9.2 Fase de Inferência (Online)

```
[USUÁRIO] → Pergunta via chatbot
       ↓
┌─────────────────────────────────────────┐
│ VALIDAÇÃO                               │
└─────────────────────────────────────────┘
       ↓ (se válida)
┌─────────────────────────────────────────┐
│ RECUPERAÇÃO MULTIMODAL                  │
├─────────────────────────────────────────┤
│ • Busca texto (k=3, Milvus)            │
│ • Busca imagem (k=3, CLIP→Milvus)      │
│ • Filtragem e ponderação                │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ GERAÇÃO                                 │
├─────────────────────────────────────────┤
│ • Contexto assemblado                   │
│ • Prompt com persona "Nanã"             │
│ • LLM Llama3-8b @ Groq                  │
└─────────────────────────────────────────┘
       ↓
[USUÁRIO] ← Resposta (texto + imagem)
```

---

## 10. Considerações de Escalabilidade

### 10.1 Escalabilidade da Ingestão

**Gargalos identificados:**

1. **Transcrição de áudio/vídeo**: Processo computacionalmente intensivo
   - **Solução**: Fila de processamento assíncrono (Celery + Redis)
   - **Estimativa**: ~1min/hora de áudio com Whisper-large

2. **Extração de embeddings**: Batch processing necessário para grandes volumes
   - **Solução**: Processamento em lote (batch_size=32)
   - **Estimativa**: ~100 chunks/segundo em CPU, ~1000/s em GPU

3. **Armazenamento no Milvus**: Linear com número de vetores
   - **Capacidade atual**: Suporta até ~10M vetores
   - **Estimativa**: ~500 PDFs de 200 páginas = ~150K chunks

### 10.2 Escalabilidade da Inferência

**Latência típica:**

```
Validação de relevância:    ~50ms
Busca no Milvus (texto):    ~20ms
Busca no Milvus (imagem):   ~15ms
Conversão base64:           ~10ms
Geração LLM (Groq):         ~1-2s
─────────────────────────────────
Total:                      ~2.1s
```

**Throughput estimado:**

- Single instance: ~30 req/min (limitado pelo LLM)
- Com load balancer: ~300 req/min (10 instâncias)

### 10.3 Otimizações Futuras

1. **Caching de embeddings**: Evitar recomputação para queries frequentes
2. **Índices Milvus avançados**: HNSW para datasets > 1M vetores
3. **Quantização**: Reduzir dimensão de 768D → 256D com PCA/AQ
4. **Reranking**: Adicionar modelo cross-encoder para refinar top-k
5. **Batch inference**: Agrupar requisições similares

---

## 11. Aspectos de Segurança e Responsabilidade

### 11.1 Validação de Conteúdo

O sistema implementa múltiplas camadas de validação:

1. **Validação de domínio**: Threshold de relevância (0.74)
2. **Filtro de imagens**: Distância + keyword matching
3. **Persona guiada**: Diretrizes antirracistas no prompt
4. **Timeout**: Limite de 10s para evitar abuse

### 11.2 Viés e Fairness

**Mitigações implementadas:**

- **Persona culturalmente sensível**: "Nanã" como guia antirracista
- **Correção de termos pejorativos**: Educação sobre "macumba", "feitiçaria"
- **Diversidade de fontes**: 15 PDFs acadêmicos e comunitários
- **Explicabilidade**: Retorno de imagens com caption para auditoria

### 11.3 Privacidade

- **Dados anonimizados**: Nenhuma informação pessoal em embeddings
- **Não persistência de queries**: Perguntas não são armazenadas
- **Base de dados controlada**: Apenas conteúdo curado pelo dashboard

---

## 12. Conclusão

Este sistema RAG multimodal representa uma arquitetura completa para democratização de conhecimento especializado, combinando:

✅ **Processamento multimodal** (texto, áudio, vídeo, imagem)  
✅ **Embeddings de alta qualidade** (BGE-768D + CLIP-512D)  
✅ **Recuperação inteligente** com ponderação adaptativa  
✅ **Geração contextualizada** via LLM otimizado (Llama3-8b)  
✅ **Safeguards de qualidade** (validação + filtragem)  
✅ **Responsabilidade cultural** (persona antirracista)  

A arquitetura é **escalável**, **modular** e preparada para expansão futura com novos tipos de conteúdo e otimizações de performance.

---

## Referências Técnicas

### Modelos

- **BAAI/bge-base-en-v1.5**: [HuggingFace](https://huggingface.co/BAAI/bge-base-en-v1.5)
- **CLIP ViT-B/32**: [OpenAI](https://github.com/openai/CLIP)
- **Llama3-8b**: [Meta AI](https://llama.meta.com/)

### Frameworks

- **LangChain**: [Documentação](https://python.langchain.com/)
- **Milvus**: [Documentação](https://milvus.io/docs)
- **Flask**: [Documentação](https://flask.palletsprojects.com/)

### Papers

- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
- *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021)
- *BGE: Multilingual Text Embeddings* (Xiao et al., 2023)

---

**Versão:** 1.0  
**Data:** Outubro de 2025  
**Projeto:** Sistema RAG Multimodal - Religiões Afro-Brasileiras

