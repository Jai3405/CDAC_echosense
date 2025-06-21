#  EchoSense: Intelligent Comment Filtering for Audio Discussions

>  **C-DAC Summer Internship Project**   
>  **Tech Stack**: FastWhisper · Gemini 1.5 Flash · SentenceTransformers · FAISS · 

---

##  Project Overview

**EchoSense** is an intelligent system designed to process public audio discussions (in Hindi), analyze associated user-submitted comments, and automatically filter them based on:

-  **Relevance**: Does the comment actually relate to the root discussion?
-  **Novelty**: Does it contribute a new point or perspective?
-  **Richness**: Does it add analytical depth?

The system enables large-scale **spoken feedback analysis**, making it ideal for government and civic discourse platforms.

---

## 🔁 End-to-End Pipeline

| Step | Stage                       | Description |
|------|-----------------------------|-------------|
| 1️⃣   | **ASR + Translation**       | Hindi audio → Text → English using FastWhisper + Gemini |
| 2️⃣   | **Text Preprocessing**      | Clean, chunk, and separate root vs. comment texts |
| 3️⃣   | **Embedding + Indexing**    | Generate vector embeddings (MPNet) & store in FAISS |
| 4️⃣   | **Relevance Filtering**     | Cosine similarity with root text to accept/reject |
| 5️⃣   | **Novelty Detection (LLM)** | Gemini LLM detects redundant vs. novel points |

---

##  Getting Started

###  Requirements

```bash
pip install -r requirements.txt
