import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "/tmp/chroma_shoes"

def load_vectordb():
    # Force clean rebuild every time (Streamlit safe)
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embed)

def fast_retriever(db):
    return db.as_retriever(search_kwargs={"k":4})

