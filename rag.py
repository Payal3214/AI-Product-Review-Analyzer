import os
import shutil

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

PERSIST_DIR = "chroma_db"


def load_vectordb(force_recreate=False):
    embed = OpenAIEmbeddings()

    if force_recreate and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embed
        )
    except ValueError:
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embed
        )

    return vectordb


def fast_retriever():
    return load_vectordb().as_retriever(search_kwargs={"k": 4})


def load_rag():
    retriever = fast_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

        
