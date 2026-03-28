"""
RAG (Retrieval-Augmented Generation) Pipeline
Uses FAISS for vector search + OpenAI for embeddings and generation
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
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
    def __init__(self, docs_dir: str = "docs", model: str = "gpt-4o-mini",
                 chunk_size: int = 500, chunk_overlap: int = 50, retriever_k: int = 4):
        self.docs_dir = Path(docs_dir)
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k

        self.vectorstore = None
        self.chain = None

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model=model, temperature=0, streaming=True)

    def _load_documents(self):
        """Load documents from folder, supporting multiple formats."""
        logging.info(f"📂 Loading documents from '{self.docs_dir}/'...")

        loaders = [
            DirectoryLoader(str(self.docs_dir), glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}),
            DirectoryLoader(str(self.docs_dir), glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(str(self.docs_dir), glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader),
        ]

        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                logging.warning(f"Skipping loader {loader}: {e}")

        if not documents:
            raise ValueError(f"No supported documents found in '{self.docs_dir}/'")

        logging.info(f"✅ Loaded {len(documents)} document(s)")
        return documents

    def load_and_index(self):
        """Load documents and build FAISS index."""
        documents = self._load_documents()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        logging.info(f"✂️  Split into {len(chunks)} chunks")

        # Build vector store
        logging.info("🔢 Building FAISS index...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logging.info("✅ Index ready!\n")

        # Build retrieval chain
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retriever_k},
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
        logging.info(f"💾 Index saved to '{path}/'")

    def load_index(self, path: str = "faiss_index"):
        """Load a previously saved FAISS index."""
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.retriever_k})
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )
        logging.info(f"✅ Index loaded from '{path}/'")
