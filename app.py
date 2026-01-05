import sys, os
sys.path.insert(0, os.getcwd())



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

import streamlit as st
from rag import load_vectordb, fast_retriever
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

st.set_page_config(page_title="AI Review Analyzer", layout="wide")
st.title("Shoes Review Analyzer")

@st.cache_resource
def load_rag():
    vectordb = load_vectordb()
    llm = HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="google/flan-t5-base"))
    return RetrievalQA.from_chain_type(llm=llm, retriever=fast_retriever(vectordb))

qa = load_rag()

query = st.text_input("Ask a product question")
if st.button("Analyze"):
    st.write(qa.run(query))
