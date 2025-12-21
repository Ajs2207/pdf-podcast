import streamlit as st
from pathlib import Path

from config.settings import UPLOAD_FOLDER
from src.utils.pdf_loader import PDFLoader
from src.utils.chunking import Chunker
from src.vectorstore.chroma_client import ChromaClient


st.set_page_config(
    page_title="PDF-Podcast (MVP)",
    layout="wide"
)

st.title("📄 PDF-Podcast – MVP")
st.caption("Upload PDFs and ingest documents (A1 stage)")


# -----------------------------
# File Upload Section
# -----------------------------
st.header("1️⃣ Upload PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    saved_files = []

    for uploaded_file in uploaded_files:
        file_path = UPLOAD_FOLDER / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_files.append(file_path)

    st.success(f"Uploaded {len(saved_files)} file(s) successfully.")


# -----------------------------
# Ingestion Section (A1 only)
# -----------------------------
st.header("2️⃣ Ingest Documents (A1)")

if st.button("Run Ingestion"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Extracting text and chunking documents..."):
            total_pages = 0
            total_chunks = 0
            vectordb = ChromaClient()

            for file_path in saved_files:
                loader = PDFLoader(file_path)
                pages = loader.load()

                chunker = Chunker()
                chunks = chunker.chunk(pages)

                vectordb.add_documents(chunks)

                total_pages += len(pages)
                total_chunks += len(chunks)

            st.success("Ingestion completed!")
            st.write(f"📄 Pages extracted: **{total_pages}**")
            st.write(f"✂️ Chunks created: **{total_chunks}**")


# -----------------------------
# Question Section (Placeholder)
# -----------------------------
st.header("3️⃣ Ask a Question (Coming Soon)")

question = st.text_input(
    "Ask a question about your documents",
    disabled=True,
    placeholder="RAG will be enabled after A2"
)

st.info("RAG-based Q&A will be enabled in the next phase (A2).")
