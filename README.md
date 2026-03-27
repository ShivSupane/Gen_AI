# 🤖 RAG LLM — Chat With Your Documents

A minimal but production-ready **Retrieval-Augmented Generation (RAG)** pipeline that lets you chat with any collection of text documents using OpenAI's LLMs.

## How It Works

```
Your .txt files  →  Chunked  →  Embedded  →  FAISS Index
                                                    ↓
User Question  →  Embedded  →  Top-k Retrieval  →  LLM  →  Answer
```

1. **Load** — reads all `.txt` files from the `docs/` folder  
2. **Chunk** — splits documents into overlapping 500-token chunks  
3. **Embed** — converts chunks to vectors using `text-embedding-3-small`  
4. **Index** — stores vectors in a local FAISS index  
5. **Query** — on each question, retrieves the top-4 relevant chunks and passes them to `gpt-4o-mini` for a grounded answer  

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/rag-llm.git
cd rag-llm
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# Edit .env and paste your OpenAI API key
```

### 3. Add your documents

Drop any `.txt` files into the `docs/` folder. A sample file is already included.

### 4. Run

```bash
python main.py
```

---

## Usage

```bash
# Interactive chat (rebuilds index each run)
python main.py

# Use a different docs folder or model
python main.py --docs-dir my_notes --model gpt-4o

# Build once, save the index, load it later (faster)
python main.py --save-index
python main.py --load-index
```

| Flag | Default | Description |
|---|---|---|
| `--docs-dir` | `docs` | Folder containing `.txt` files |
| `--model` | `gpt-4o-mini` | OpenAI chat model |
| `--save-index` | — | Save FAISS index after building |
| `--load-index` | — | Skip rebuilding, load saved index |
| `--index-path` | `faiss_index` | Path for saved index |

---

## Project Structure

```
rag-llm/
├── main.py              # CLI entry point
├── src/
│   └── rag.py           # RAGPipeline class
├── docs/
│   └── ai_overview.txt  # Sample document
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (local, no server needed) |
| Orchestration | LangChain |

---

## Extending the Project

- **More file types** — add `PyPDFLoader` or `UnstructuredMarkdownLoader`  
- **Better UI** — wrap with Streamlit or FastAPI  
- **Persistent store** — swap FAISS for Chroma or Pinecone  
- **Reranking** — add a cross-encoder to improve retrieval quality  

---

## License

MIT
