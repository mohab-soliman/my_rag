import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F0F0F;
    color: #E8E2D9;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 860px; }

.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid #222;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #E8E2D9;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero p { color: #666; font-size: 0.95rem; font-weight: 300; margin-top: 0.5rem; letter-spacing: 0.5px; }
.hero .dot { color: #C9A96E; }

[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0.5rem 0 !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background: #1A1A1A; border: 1px solid #2A2A2A;
    border-radius: 16px 16px 4px 16px; padding: 0.9rem 1.2rem; color: #E8E2D9;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
    background: #161410; border: 1px solid #C9A96E33;
    border-radius: 16px 16px 16px 4px; padding: 0.9rem 1.2rem; color: #E8E2D9;
}
[data-testid="stChatInput"] { border-top: 1px solid #222 !important; padding-top: 1rem; }
[data-testid="stChatInputTextArea"] {
    background: #1A1A1A !important; border: 1px solid #2A2A2A !important;
    border-radius: 12px !important; color: #E8E2D9 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: #C9A96E !important; box-shadow: 0 0 0 2px #C9A96E22 !important;
}
[data-testid="stExpander"] {
    background: #111 !important; border: 1px solid #222 !important;
    border-radius: 10px !important; margin-top: 0.5rem;
}
[data-testid="stExpander"] summary { color: #666 !important; font-size: 0.85rem !important; }
hr { border-color: #222 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0F0F0F; }
::-webkit-scrollbar-thumb { background: #2A2A2A; border-radius: 2px; }
</style>

<div class="hero">
    <h1>📚 Library Assistant<span class="dot">.</span></h1>
    <p>Ask anything about your library collection</p>
</div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Library Assistant", page_icon="📚", layout="centered")


# ── APA Citation Helper ──────────────────────────────────────
def build_apa_citation(metadata):
    author    = metadata.get("author", "Unknown Author")
    year      = metadata.get("year", "n.d.")
    title     = metadata.get("title", "Untitled")
    publisher = metadata.get("publisher", "")
    page      = metadata.get("page", None)
    citation  = f"{author}. ({year}). *{title}*."
    if publisher:
        citation += f" {publisher}."
    if page:
        citation += f" p. {int(page) + 1}"
    return citation


# ── Build RAG Chain ──────────────────────────────────────────
@st.cache_resource
def build_rag_chain():
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Chroma(
        persist_directory="vector_DB",
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    system_prompt = (
        "You are a professional Library Assistant. "
        "Use the retrieved context about books to answer the user's question. "
        "If the answer is not in the context, say that you don't know. "
        "\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# ── Chat History ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Main App ─────────────────────────────────────────────────
rag_chain = build_rag_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask about a book, author, or topic...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Searching..."):
        result = rag_chain.invoke({"input": query})

    answer = result["answer"]
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)

        # APA Citations
        st.markdown("---")
        st.markdown("<small>📎 <b>References</b></small>", unsafe_allow_html=True)
        seen_citations = set()
        for doc in result["context"]:
            citation = build_apa_citation(doc.metadata)
            if citation not in seen_citations:
                seen_citations.add(citation)
                st.caption(citation)

        # Source Chunks
        with st.expander("View source chunks"):
            for i, doc in enumerate(result["context"], 1):
                st.caption(f"Chunk {i}")
                st.write(doc.page_content)
                st.divider()
