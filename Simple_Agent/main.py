"""
CLI entry point — run the RAG chatbot interactively.

Usage:
    python main.py                  # interactive chat (rebuilds index)
    python main.py --load-index     # load saved index, skip re-indexing
    python main.py --save-index     # rebuild and save index for later
"""

import argparse
from src.rag import RAGPipeline


def parse_args():
    p = argparse.ArgumentParser(description="RAG LLM — chat with your documents")
    p.add_argument("--docs-dir", default="docs", help="Folder with .txt documents")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    p.add_argument("--load-index", action="store_true", help="Load saved FAISS index")
    p.add_argument("--save-index", action="store_true", help="Save FAISS index after building")
    p.add_argument("--index-path", default="faiss_index", help="Path for saved index")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  🤖  RAG LLM — Chat With Your Documents")
    print("=" * 55)

    rag = RAGPipeline(docs_dir=args.docs_dir, model=args.model)

    if args.load_index:
        rag.load_index(args.index_path)
    else:
        rag.load_and_index()
        if args.save_index:
            rag.save_index(args.index_path)

    print("Type your question below (or 'quit' to exit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("👋 Bye!")
            break

        print("🤔 Thinking...")
        answer = rag.ask(question)
        print(f"\n🤖 {answer}\n")
        print("-" * 45)


if __name__ == "__main__":
    main()
