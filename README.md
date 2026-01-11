# Open Domain Question Answering (ODQA)

## Project Overview

This project presents a **Retrieval-Augmented Question Answering (QA) system** built around a fine-tuned **BERT-based reader**. The core idea is to answer natural language questions by combining **information retrieval** with **extractive question answering**, enabling the system to scale beyond a single document while maintaining accurate answer extraction.

The project focuses on **designing, training, and evaluating** each component of the QA pipeline rather than deployment.

---

## System Architecture

The project implements a **two-stage Open-Domain Question Answering (ODQA) architecture** based on the **Retriever–Reader paradigm**. The system is designed to efficiently search a large corpus of documents and extract precise answers using deep semantic representations.

At a high level, the architecture is composed of:

1. A **Dense Retrieval module (DPR)** for semantic passage selection
2. A **Vector indexing and similarity search layer (FAISS)**
3. A **Reader module (BERT + LoRA)** for extractive answer prediction

---

## Global Architecture Overview

```
User Question
      │
      ▼
Question Encoder (DPR)
      │  → Dense Question Embedding (768-d)
      ▼
Cosine Similarity Search (FAISS)
      │
      ▼
Top-k Relevant Passages
      │
      ▼
Reader (BERT for QA + LoRA)
      │
      ▼
Answer Span Scoring & Selection
      │
      ▼
Final Answer
```

This modular design clearly separates **retrieval**, **semantic matching**, and **answer comprehension**, ensuring scalability and interpretability.

---

## Dense Passage Retrieval (DPR)

The retrieval module uses **Dense Passage Retrieval (DPR)** to perform semantic search over a large corpus.

### Dual-Encoder Architecture

* **Question Encoder**: Encodes the user question into a dense vector
* **Context Encoder**: Encodes each passage in the corpus into dense vectors

Both encoders:

* Are BERT-based
* Produce **768-dimensional embeddings**
* Project questions and passages into a **shared semantic space**

This allows semantic similarity to be computed directly between questions and passages.

---

## Cosine Similarity for Semantic Matching

To measure relevance between a question and a passage, the system uses **cosine similarity**.

### Embedding Normalization

All embeddings are **L2-normalized**, ensuring each vector has unit length:

* This removes the influence of vector magnitude
* Similarity depends only on the **angle between vectors**

### Cosine Similarity Definition

For a question embedding (q) and a passage embedding (p):

```
cosine_similarity(q, p) = (q · p) / (||q|| · ||p||)
```

After L2 normalization:

```
||q|| = ||p|| = 1
⇒ cosine_similarity(q, p) = q · p
```

This allows cosine similarity to be computed efficiently using a **dot product**, which is directly supported by FAISS.

---

## FAISS Vector Indexing and Search

To enable fast similarity search at scale, the system relies on **FAISS (Facebook AI Similarity Search)**.

### Index Design

* **Index type**: `IndexFlatIP` (Inner Product)
* **Similarity metric**: Inner product ≡ cosine similarity (after normalization)

### Index Construction

* Passages are encoded offline using the DPR context encoder
* Embeddings are L2-normalized
* Vectors are stored in the FAISS index

### Query-Time Retrieval

1. Encode the question using the DPR question encoder
2. Normalize the question embedding
3. Perform top-*k* nearest neighbor search in FAISS
4. Retrieve the most semantically similar passages

This design ensures **accurate and efficient semantic retrieval**, even with large corpora.

---

## Reader Module: BERT for Extractive QA

The reader module performs **fine-grained comprehension** of retrieved passages.

### Reader Architecture

* Based on **BERT for Question Answering**
* Operates in an **extractive setting**
* Input format: `[CLS] Question [SEP] Passage [SEP]`

### Predictions

For each passage, the model predicts:

* Start token probability
* End token probability
* Confidence score for the extracted span

Multiple candidate answers are generated—one per retrieved passage.

---

## Parameter-Efficient Fine-Tuning with LoRA

To fine-tune the reader efficiently, the project uses **LoRA (Low-Rank Adaptation)**.

### LoRA Mechanism

* Original BERT weights are **frozen**
* Small trainable low-rank matrices are injected into attention layers
* Only LoRA parameters are updated during training

### Advantages

* Significant reduction in trainable parameters
* Lower GPU memory usage
* Faster convergence
* Performance comparable to full fine-tuning

This makes the approach well-suited for academic and resource-constrained environments.

---

## End-to-End Inference Flow

The complete inference pipeline follows these steps:

1. User question is encoded by the DPR question encoder
2. Cosine similarity search retrieves top-*k* passages via FAISS
3. Each passage is processed by the BERT reader
4. Answer spans are scored across passages
5. The highest-confidence answer is selected as the final output

---

## What We Built

* A **dense semantic retriever** using DPR
* A **cosine-similarity-based search engine** with FAISS
* An **extractive QA reader** based on BERT
* A **parameter-efficient fine-tuning strategy** using LoRA
* A **scalable and modular ODQA system**
