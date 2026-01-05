import streamlit as st
from rag import load_rag

st.set_page_config(page_title="AI Review Analyzer", layout="wide")
st.title("Shoes Review Analyzer")

@st.cache_resource
def load_qa():
    return load_rag()   # use the safe RAG loader from rag.py

qa = load_qa()

query = st.text_input("Ask a product question")

if st.button("Analyze") and query:
    with st.spinner("Analyzing reviews..."):
        result = qa.run(query)
        st.success("Done!")
        st.write(result)


