import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Page config

st.set_page_config(
    page_title="ArmRAG",
    layout="wide"
)


# Load embeddings (query-time)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Load FAISS vectors

vectorstore = FAISS.load_local(
    "vectors",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# Groq LLM

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)


# Prompt (STRICT RAG)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict armwrestling knowledge assistant. "
            "Answer ONLY using the provided context. "
            "If the answer is not in the context, say it is not mentioned."
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ]
)


# Document chain (LLM + prompt)

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)


# Retrieval chain (retriever + doc chain)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)


# UI

st.title("ArmRAG — Armwrestling Knowledge Assistant")
st.markdown(
    "Ask questions about **armwrestling techniques, training, injuries, rules, and competition strategy**."
)

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        result = rag_chain.invoke({"input": query})

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for doc in result["context"]:
        page = doc.metadata.get("page", "?")
        st.markdown(f"- **Page {page}**: {doc.page_content[:200]}…")
