# 🤖 Shiv GenAI RAG — Chat With Your Documents

A powerful **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents and interact with them using **LLMs (Large Language Models)**.

Built using **Streamlit + FAISS + HuggingFace Embeddings + Groq API**, this app delivers fast, intelligent, and context-aware responses.

---

## 🚀 Features

* 📂 Upload your own `.txt` documents
* 🔍 Intelligent document search using **FAISS vector database**
* 🧠 Context-aware answers using **RAG pipeline**
* ⚡ Ultra-fast responses powered by **Groq LLMs**
* 💬 Chat-style interface (like ChatGPT)
* 🌐 Ready for deployment (Streamlit Cloud / Hugging Face)

---

## 🏗️ Tech Stack

* **Frontend/UI**: Streamlit
* **LLM Provider**: Groq (LLaMA 3.1 / Mixtral)
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
* **Vector DB**: FAISS
* **Framework**: LangChain

---

## 🧠 How It Works (RAG Pipeline)

1. 📄 Upload documents
2. ✂️ Split text into chunks
3. 🔢 Convert chunks into embeddings
4. 📦 Store embeddings in FAISS
5. 🔎 Retrieve relevant chunks based on query
6. 🤖 Send context + question to LLM
7. 💡 Generate accurate answer

---

## 📁 Project Structure

```
rag-based-llm-application/
│
├── app.py               # Streamlit UI
├── main.py              # CLI version
├── requirements.txt
├── README.md
│
├── docs/                # Uploaded documents
│
└── src/
    └── rag.py           # RAG pipeline logic
```

---

## ⚙️ Installation (Local Setup)

```bash
# Clone the repository
git clone https://github.com/your-username/rag-based-llm-application.git

cd rag-based-llm-application

# Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Set API Key

### Option 1: Environment Variable

```bash
set GROQ_API_KEY=your_api_key_here
```

### Option 2: Streamlit Secrets (Recommended for deployment)

```
GROQ_API_KEY = "your_api_key_here"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🌐 Deployment

### 🚀 Streamlit Cloud

1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Add `GROQ_API_KEY` in Secrets
4. Deploy

---

### 🤗 Hugging Face Spaces

1. Create new Space (Streamlit)
2. Upload project files
3. Add API key in **Settings → Secrets**
4. Run automatically

---

## 📸 Screenshots

> Add screenshots of your UI here for better presentation

---

## 🎯 Use Cases

* 📚 Study assistant
* 📄 Resume analyzer
* 🏢 Document QA system
* 📊 Knowledge base chatbot
* 🧑‍💻 Developer documentation assistant

---

## ⚠️ Limitations

* Supports `.txt` files only (can be extended)
* Requires internet for LLM API
* Performance depends on document size

---

## 🔮 Future Improvements

* 📄 Support PDF, DOCX
* 🧠 Memory-based conversations
* 🌍 Multi-language support
* 📊 Analytics dashboard
* 🧑‍🤝‍🧑 Multi-user system

---

## 👨‍💻 Author

**Shiv Supane**
🚀 Passionate about AI, Cloud & Scalable Systems

---

## ⭐ Show Your Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 📢 Share it

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

> 💡 “Build AI that doesn’t just generate — but understands.”
