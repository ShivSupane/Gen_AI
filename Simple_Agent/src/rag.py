"""
RAG (Retrieval-Augmented Generation) Pipeline
Uses FAISS for vector search + OpenAI for embeddings and generation
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()


# ── Prompt Template ──────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""",
)


# ── RAG Pipeline ─────────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self, docs_dir: str = "docs", model: str = "gpt-4o-mini"):
        self.docs_dir = Path(docs_dir)
        self.model = model
        self.vectorstore = None
        self.chain = None

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model=model, temperature=0)

    def load_and_index(self):
        """Load documents from docs/ folder and build FAISS index."""
        print(f"📂 Loading documents from '{self.docs_dir}/'...")

        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()

        if not documents:
            raise ValueError(f"No .txt files found in '{self.docs_dir}/'")

        print(f"✅ Loaded {len(documents)} document(s)")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        print(f"✂️  Split into {len(chunks)} chunks")

        # Build vector store
        print("🔢 Building FAISS index...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("✅ Index ready!\n")

        # Build retrieval chain
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )

    def ask(self, question: str) -> str:
        """Ask a question against the indexed documents."""
        if self.chain is None:
            raise RuntimeError("Call load_and_index() first.")
        result = self.chain.invoke({"query": question})
        return result["result"]

    def save_index(self, path: str = "faiss_index"):
        """Save the FAISS index to disk."""
        self.vectorstore.save_local(path)
        print(f"💾 Index saved to '{path}/'")

    def load_index(self, path: str = "faiss_index"):
        """Load a previously saved FAISS index."""
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )
        print(f"✅ Index loaded from '{path}/'")
