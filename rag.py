import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.retrievers import RetrievalQA  # <- updated
from langchain_openai import ChatOpenAI




PERSIST_DIR = "chroma_db"


def load_vectordb(force_recreate=False):
    if force_recreate and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    loader = PyPDFLoader("data/reviews.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    return vectordb


def load_rag():
    retriever = load_vectordb().as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
