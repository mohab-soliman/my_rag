import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="AGRIRA - Intelligent Agriculture RAG", page_icon="🌱")

# ـ API  
os.environ["GOOGLE_API_KEY"] = "AIzaSyCl7w9q_RO7yeU5HAVaTQST1_AbTPJST3o"

try:
    st.image("image_deb840dc.jpg", width=150) 
except:
    st.title("🌱 AGRIRA")

st.header("AGRIRA: Intelligent Agriculture RAG")
st.write("مرحباً بك في مساعدك الزراعي الذكي. اسأل أي سؤال بناءً على مستنداتك.")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def build_rag_chain():
    # تحميل  Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(
        persist_directory="MY VICTOR DB", 
        embedding_function=embedding_model
    )
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    system_prompt = (
        "You are AGRIRA, a professional Agriculture Assistant. "
        "Use the retrieved context about agriculture to answer the user's question. "
        "If the answer is not in the context, say that you don't know. "
        "\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

rag_chain = build_rag_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask about agriculture topics...")


if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("AGRIRA is thinking..."):
        result = rag_chain.invoke({"input": query}) 
        answer = result["answer"]
        
        with st.chat_message("assistant"):
            st.markdown(answer)

            st.markdown("---")
            st.markdown("<small>📚 <b>References</b></small>", unsafe_allow_html=True)
            seen_citations = set()
            for doc in result["context"]:
                citation = build_apa_citation(doc.metadata)
                if citation not in seen_citations:
                    seen_citations.add(citation)
                    st.caption(citation)







