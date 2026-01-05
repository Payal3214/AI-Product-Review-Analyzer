import os
import shutil

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

PERSIST_DIR = "chroma_db"


# ---------------- VECTOR STORE LOADER ----------------

def load_vectordb(force_recreate: bool = False):
    embed = OpenAIEmbeddings()

    if force_recreate and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print("[INFO] Existing Chroma DB deleted")

    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embed
        )
        print("[INFO] Chroma DB loaded successfully")
    except ValueError:
        print("[WARNING] Embedding conflict detected. Rebuilding DB...")
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embed
        )
        print("[INFO] New Chroma DB created")

    return vectordb


# ---------------- RETRIEVER ----------------

def fast_retriever():
    vectordb = load_vectordb()
    return vectordb.as_retriever(search_kwargs={"k": 4})


# ---------------- QA CHAIN ----------------

def load_rag():
    llm = ChatOpenAI(temperature=0)
    retriever = fast_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa

        

