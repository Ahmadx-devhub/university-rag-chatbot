from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
DB_DIR = "./chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def ingest_documents():

    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
        return

    for pdf in pdf_files:

        print(f"Loading: {pdf.name}")

        loader = PyPDFLoader(str(pdf))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf.name

        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vector_store = Chroma(
        collection_name="university_documents",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    vector_store.add_documents(documents=chunks)

    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print("Documents successfully stored in ChromaDB!")


if __name__ == "__main__":
    ingest_documents()