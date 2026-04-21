import streamlit as st
import fitz
import os
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="PDF Q&A System", layout="wide")
st.title("RAG Document Intelligence System")
st.caption("Upload PDFs and ask questions about them")

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
client = Groq()

def load_pdfs(uploaded_files):
    documents = []
    for file in uploaded_files:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        documents.append({"name": file.name, "text": text})
    return documents

def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    all_chunks = []
    all_metadatas = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        all_chunks.extend(chunks)
        all_metadatas.extend([{"source": doc["name"]}] * len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(
        all_chunks,
        embeddings,
        metadatas=all_metadatas
    )
    return vectorstore

def ask_question(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata["source"] for d in docs]))

    prompt = f"""Use only the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return answer, sources

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        for f in uploaded_files:
            st.write(f"- {f.name}")

# Main area — FIX 1: vectorstore stored in session_state so it persists
if uploaded_files:
    if "vectorstore" not in st.session_state or st.session_state.get("files") != [f.name for f in uploaded_files]:
        with st.spinner("Processing PDFs... please wait"):
            documents = load_pdfs(uploaded_files)
            st.session_state.vectorstore = create_vectorstore(documents)
            st.session_state.files = [f.name for f in uploaded_files]
        st.success("PDFs processed! Ask your questions below.")
    else:
        st.success("PDFs ready! Ask your questions below.")

    # FIX 2: question input inside the uploaded_files block
    question = st.text_input("Ask a question about your PDFs")

    # FIX 3: answer block inside uploaded_files block
    if question:
        with st.spinner("Thinking..."):
            answer, sources = ask_question(question, st.session_state.vectorstore)
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("### Sources")
        for s in sources:
            st.caption(f"- {s}")

else:
    st.info("Please upload at least one PDF from the sidebar to get started.")