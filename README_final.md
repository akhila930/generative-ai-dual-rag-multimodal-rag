# Generative AI End-Term — Dual RAG and Multimodal RAG

## Overview

This project implements the STAI345 Generative AI end-term requirement using the Vijaybhoomi University **Research Incubation Handbook**.

The project has two phases:

- **Phase 1:** Two local text-only RAG pipelines using the same handbook, embedding model and FAISS vector store:
  - SmolLM2-135M
  - Llama 3.2-1B-Instruct Q4_K_M
- **Phase 2:** Multimodal RAG using retrieved handbook text + handbook page images with SmolVLM-256M.

A Streamlit application provides an interactive demonstration.

---

## 1. Architecture

### Phase 1

```text
University Handbook PDF
        |
        v
Document Processing
        |
        v
Page Text + Page Numbers
        |
        v
Embedding Model
        |
        v
FAISS Vector Store
        |
        v
User Question
        |
        v
Top-K Retrieval
        |
        +-----------------------+
        |                       |
        v                       v
 SmolLM2-135M          Llama 3.2-1B Q4_K_M
        |                       |
        +-----------+-----------+
                    |
                    v
                Answer
                    |
                    v
               Evaluation
```

### Phase 2

```text
User Question
      |
      v
Shared FAISS Retrieval
      |
      +--------------------+
      |                    |
      v                    v
Retrieved Text       Retrieved Page Images
      |                    |
      +---------+----------+
                |
                v
        SmolVLM-256M
                |
                v
        Multimodal Answer
                |
                v
            Evaluation
```

---

## 2. Directory Structure

```text
generative_ai_endterm_rag/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── document.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── rag.py
│   ├── evaluation.py
│   ├── multimodal.py
│   └── multimodal_evaluation.py
├── data/
│   ├── university_handbook.pdf
│   └── evaluation_questions.json
├── notebooks/
│   ├── 01_document_processing.ipynb
│   ├── 02_dual_rag.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_multimodal_rag.ipynb
├── diagrams/
│   ├── dual_rag_architecture.mmd
│   └── multimodal_rag_architecture.mmd
├── results/
├── requirements.txt
├── README.md
├── setup_linux_mac.sh
└── setup_windows.bat
```

---

## 3. Models

| Component | Model / Tool |
|---|---|
| Text model 1 | SmolLM2-135M |
| Text model 2 | Llama 3.2-1B-Instruct Q4_K_M |
| Multimodal model | SmolVLM-256M-Instruct Q8_0 |
| Vector database | FAISS |
| Embeddings | Sentence-Transformers |
| UI | Streamlit |
| Multimodal runtime | llama.cpp |

The Llama model is quantized to Q4_K_M because the full BF16 model was too memory-intensive for the available CPU/RAM environment.

---

## 4. Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
```

### Multimodal runtime

Install the llama.cpp CLI using the official llama.cpp/llama installer available for the environment.

The multimodal pipeline uses:

```text
ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0
```

The first multimodal run downloads/caches the model and multimodal projector.

---

## 5. Prepare the Handbook

Process the PDF:

```bash
python -m src.document
```

Build the vector store:

```bash
python -m src.vector_store
```

The handbook contains 15 pages.

---

## 6. Run Phase 1

### SmolLM2

```bash
python -m src.rag --model smollm
```

### Llama 3.2

```bash
python -m src.rag --model llama
```

Both pipelines use the same retrieval system.

---

## 7. Phase 1 Evaluation

Run:

```bash
python -m src.evaluation
```

The evaluation produces:

```text
results/smollm_evaluation.csv
results/smollm_evaluation.json
results/smollm_summary.json

results/llama_evaluation.csv
results/llama_evaluation.json
results/llama_summary.json
```

### Metrics

The project records:

- Context Precision
- Context Recall
- Mean Reciprocal Rank (MRR)
- Answer Similarity
- Keyword Coverage
- Answer Correctness Proxy
- Answer Relevancy Proxy
- Groundedness Proxy
- Latency

The correctness, relevancy and groundedness measures are lightweight project-specific proxies. They should not be interpreted as a full RAGAS or human-judge evaluation.

---

## 8. Phase 1 Observations

For the representative question:

> What is the purpose of the Research Incubation program?

both models generated a correct answer.

The same retrieved pages were:

```text
[1, 3, 15]
```

Representative latency:

```text
SmolLM2-135M: approximately 14.42 seconds
Llama 3.2-1B: approximately 21.85 seconds
```

The engineering trade-off is:

- SmolLM2-135M: lower resource usage and lower latency.
- Llama 3.2-1B: stronger quality-oriented generation and better suitability when answer quality is prioritized.

For this experiment, Llama 3.2-1B is treated as the preferred quality-oriented model, while SmolLM2 is the lightweight baseline.

---

## 9. Phase 2 — Multimodal RAG

Run the multimodal pipeline:

```bash
python -m src.multimodal
```

Run the two-case multimodal evaluation:

```bash
python -m src.multimodal_evaluation
```

Result:

```text
results/multimodal_evaluation.json
```

The multimodal pipeline:

1. Retrieves handbook pages.
2. Uses the existing page number/image mapping.
3. Loads the corresponding page images.
4. Combines retrieved text and images.
5. Sends them to SmolVLM-256M.
6. Returns the answer, retrieved pages, images and latency.

---

## 10. Multimodal Findings

Two representative tests were used.

### MM01

Question:

```text
What information is shown on page 3 of the Research Incubation Handbook?
```

Retrieved pages:

```text
[1, 15]
```

Keyword coverage:

```text
0.0
```

Latency:

```text
27.206 seconds
```

The relevant page was not retrieved, so the VLM received the wrong visual context.

### MM02

Question:

```text
What grades and grade points are specified for the Research Incubation course?
```

Retrieved pages:

```text
[6, 11]
```

Keyword coverage:

```text
0.7
```

Latency:

```text
19.108 seconds
```

The relevant page was present in the retrieved results, but the lightweight VLM still struggled with structured numerical information.

A top-3 retrieval version was subsequently tested:

```text
[6, 11, 3]
```

This successfully loaded all three page images, but the final VLM answer was still unreliable. The top-3 run took approximately 32.5 seconds.

The main finding is:

> Multimodal input does not automatically solve retrieval or structured-information extraction problems.

---

## 11. Streamlit Application

Start the UI:

```bash
streamlit run app/streamlit_app.py
```

The application contains three tabs:

### RAG Chat

Select:

- SmolLM2-135M
- Llama 3.2-1B Q4_K_M

and ask questions about the handbook.

### Evaluation

Displays the saved Phase 1 evaluation results and model comparison.

### Multimodal RAG

Displays:

- user question
- generated answer
- latency
- retrieved pages
- retrieved handbook page images
- retrieved text context

---

## 12. Failure Analysis

### Retrieval noise

Page-level retrieval can return irrelevant pages.

**Cause:** The handbook is indexed at page level and similar introductory/footer content occurs across pages.

**Possible improvement:** smaller semantic chunks, reranking, or hybrid retrieval.

### Lightweight VLM limitations

SmolVLM-256M can identify general page content but can fail on tables and numerical relationships.

**Possible improvement:** a stronger VLM or a dedicated OCR/table extraction stage.

### Multimodal latency

Adding several images increases processing time.

**Possible improvement:** image caching, fewer retrieved images, lower image resolution, or a persistent model server.

### Hardware limitation

The full BF16 Llama model was too memory-intensive for the available environment.

**Solution used:** Q4_K_M GGUF through llama.cpp.

---

## 13. Reproducibility

The main execution order is:

```bash
python -m src.document
python -m src.vector_store

python -m src.rag --model smollm
python -m src.rag --model llama

python -m src.evaluation

python -m src.multimodal
python -m src.multimodal_evaluation

streamlit run app/streamlit_app.py
```

---

## 14. Main Conclusion

The project demonstrates a complete local RAG workflow and its extension to Multimodal RAG.

The main engineering conclusions are:

1. Retrieval quality is a major determinant of final answer quality.
2. A larger local model can provide stronger answer generation but usually requires more compute and latency.
3. Quantization makes a 1B-class model practical on constrained hardware.
4. Multimodal RAG adds useful visual context but introduces additional model and latency constraints.
5. Lightweight VLMs can still fail on structured numerical information.
6. A robust production system should combine retrieval, generation and validation rather than relying only on the generative model.

---

## 15. Important Limitation

This project is an end-term engineering experiment using a small local evaluation set and lightweight automated metrics. The results are intended to compare the implemented systems under the same experimental setup, not to make general claims about the models on all RAG tasks.
