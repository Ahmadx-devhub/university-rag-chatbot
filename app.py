
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()

st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "university_documents"




def get_secret(key):


    try:
        value = st.secrets.get(key)
        if value:
            return value
    except Exception:
        pass

   
    return os.getenv(key)


@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )




@st.cache_resource
def get_vector_store():

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

    document_count = vector_store._collection.count()

    if document_count > 0:

        return vector_store

    st.info("First-time setup: Building university knowledge base...")

  
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found inside {DATA_DIR}"
        )

    documents = []


    for pdf_file in pdf_files:

        loader = PyPDFLoader(str(pdf_file))

        pdf_documents = loader.load()

        for document in pdf_documents:

            document.metadata["source"] = pdf_file.name

        documents.extend(pdf_documents)

    if not documents:

        raise ValueError("No readable content found in PDF files.")

   
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    if not chunks:

        raise ValueError("No document chunks were generated.")

 
    vector_store.add_documents(chunks)

    return vector_store



@st.cache_resource
def get_llm():

    api_key = get_secret("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
        )

    model_name = get_secret("GROQ_MODEL") or "openai/gpt-oss-120b"

    return ChatGroq(
        model=model_name,
        temperature=0.2,
        groq_api_key=api_key
    )




if "messages" not in st.session_state:

    st.session_state.messages = []



st.title("🎓 University Helpdesk Chatbot")

st.caption("Ask anything about university documents.")



try:

    vector_store = get_vector_store()

except Exception as e:

    st.error(f"Knowledge base initialization failed: {e}")

    st.stop()



with st.sidebar:

    st.title("🎓 University Assistant")

    st.success("Knowledge Base Connected")

    st.write(
        f"Documents chunks: {vector_store._collection.count()}"
    )

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()



for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])




question = st.chat_input("Ask your university-related question...")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner("Finding your answer..."):

            try:

                llm = get_llm()

               
                docs = vector_store.similarity_search(
                    question,
                    k=4
                )

              
                context = ""

                for i, doc in enumerate(docs):

                    source = doc.metadata.get("source", "Unknown")

                    page = doc.metadata.get("page", 0) + 1

                    context += f"""
Source [{i + 1}]
File: {source}
Page: {page}
Content: {doc.page_content}

"""

                
                history = ""

                for msg in st.session_state.messages[-7:-1]:

                    history += f"""
{msg['role']}: {msg['content']}
"""

            
                prompt = f"""
You are a University Helpdesk AI Assistant.

Your job is to answer university-related questions.

IMPORTANT RULES:

1. Answer using only the provided university documents.
2. Do not invent information.
3. If the answer is not available in the context,
   clearly say that you could not find the information
   in the available university documents.
4. Give clear and simple answers.
5. Use previous conversation only to understand follow-up questions.

Previous Conversation:
{history}

Retrieved University Documents:
{context}

User Question:
{question}

Answer:
"""

            
                response = llm.invoke(prompt)

                answer = response.content

                if not isinstance(answer, str):
                    answer = str(answer)

                st.markdown(answer)

              
                with st.expander("📚 View Sources"):

                    if docs:

                        for doc in docs:

                            source = doc.metadata.get(
                                "source",
                                "Unknown"
                            )

                            page = doc.metadata.get(
                                "page",
                                0
                            ) + 1

                            st.write(
                                f"📄 {source} | Page {page}"
                            )

                    else:

                        st.write("No sources found.")

            except Exception as e:

                answer = f"Something went wrong: {e}"

                st.error(answer)


   
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
