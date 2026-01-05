from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectordb():
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="models/chroma_shoes", embedding_function=embed)

def fast_retriever(db):
    return db.as_retriever(search_kwargs={"k":4})
