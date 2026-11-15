import os
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓"
)

DB_DIR = "./chroma_db"


@st.cache_resource
def get_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name="university_documents",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )



@st.cache_resource
def get_llm():

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )



if "messages" not in st.session_state:
    st.session_state.messages = []



st.title("🎓 University Helpdesk Chatbot")

st.caption("Ask anything about university documents.")


vector_store = get_vector_store()

if vector_store._collection.count() == 0:

    st.error("Knowledge base is empty. Please contact the administrator.")

    st.stop()



if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()



for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])



question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Finding your answer..."):

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
                Source [{i+1}]
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

            Answer using only the provided university documents.

            If the answer is not available in the context,
            clearly say you don't know.

            Previous conversation:
            {history}

            Retrieved context:
            {context}

            User question:
            {question}

            Answer:
            """

            response = llm.invoke(prompt)

            answer = response.content

            st.markdown(answer)

           
            with st.expander("📚 View Sources"):

                for doc in docs:

                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", 0) + 1

                    st.write(f"📄 {source} | Page {page}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })