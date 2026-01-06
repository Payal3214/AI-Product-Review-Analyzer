import streamlit as st
from rag import load_rag

st.set_page_config(page_title="Shoes Review Analyzer")

qa = load_rag()

st.title("Shoes Review Analyzer")

query = st.text_input("Ask something about your product reviews:")

if query:
    result = qa.invoke({"query": query})
    st.write(result["result"])



