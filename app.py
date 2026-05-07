import streamlit as st
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Library Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── MODERN UI / CSS ─────────────────────────────────────────
# Note: Google Fonts import works better in the <head>, but this fallback works for most cases
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0B1120;
    --card: rgba(15, 23, 42, 0.72);
    --border: rgba(255,255,255,0.08);
    --text: #F8FAFC;
    --muted: #94A3B8;
    --accent: #7C3AED;
    --accent-2: #06B6D4;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at top left, #111827 0%, #0B1120 45%, #020617 100%);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(circle at top left, #111827 0%, #0B1120 45%, #020617 100%);
}

.block-container {
    max-width: 920px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HERO SECTION */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.5rem;
    margin-bottom: 2rem;
    border-radius: 28px;
    border: 1px solid var(--border);
    background: linear-gradient(135deg,
        rgba(124,58,237,0.22),
        rgba(6,182,212,0.12),
        rgba(15,23,42,0.95));
    backdrop-filter: blur(18px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
}

.hero::before {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    background: rgba(124,58,237,0.18);
    border-radius: 50%;
    top: -140px;
    right: -120px;
    filter: blur(60px);
}

.hero-grid {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    position: relative;
    z-index: 2;
}

.hero-text h1 {
    margin: 0;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -1.5px;
    color: white;
}

.hero-text p {
    margin-top: 1rem;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
    max-width: 560px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .45rem .85rem;
    margin-bottom: 1.2rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    color: #CBD5E1;
    font-size: .82rem;
    font-weight: 500;
}

.hero-logo {
    width: 120px;
    height: 120px;
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    overflow: hidden;
    font-size: 3rem;
}

/* CHAT BUBBLES */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: #111827 !important;
}

[data-testid="stChatMessage"] .stMarkdown {
    padding: 1rem 1.2rem;
    border-radius: 20px;
    line-height: 1.7;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background: linear-gradient(135deg, rgba(124,58,237,.22), rgba(6,182,212,.15));
    border: 1px solid rgba(124,58,237,.25);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(255,255,255,.06);
}

/* INPUT */
[data-testid="stChatInputTextArea"] {
    background: rgba(15,23,42,.9) !important;
    color: white !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    padding: .8rem 1rem !important;
}

[data-testid="stChatInputTextArea"]:focus {
    border: 1px solid rgba(124,58,237,.7) !important;
    box-shadow: 0 0 0 4px rgba(124,58,237,.15) !important;
}

/* REFERENCES */
.reference-card {
    margin-top: 1rem;
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: rgba(15,23,42,.65);
    border: 1px solid rgba(255,255,255,.06);
}

[data-testid="stExpander"] {
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,.06) !important;
    background: rgba(15,23,42,.55) !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 20px; }
</style>

<div class="hero">
    <div class="hero-grid">
        <div class="hero-text">
            <div class="badge">✨ AI Powered Library Retrieval System</div>
            <h1>Library Assistant</h1>
            <p>Search your library collection intelligently using Retrieval-Augmented Generation (RAG), semantic search, and Gemini AI.</p>
        </div>
        <div class="hero-logo">📚</div>
    </div>
</div>
""", unsafe_allow_html=True)

# APA Citation Helper
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

# Build RAG Chain
@st.cache_resource
def build_rag_chain():
    # 1. Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 2. Vector Store (Ensure directory exists)
    db_path = "vector_DB"
    Path(db_path).mkdir(parents=True, exist_ok=True)
    
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )
    
    # 3. LLM Setup with API Key Check
    api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("❌ Google API Key not found! Please add it to `.streamlit/secrets.toml`")
        st.stop()
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Fixed: gemini-2.5-flash is not available yet
        temperature=0.2,
        google_api_key=api_key
    )
    
    # 4. Prompt & Chain
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
try:
    rag_chain = build_rag_chain()
except Exception as e:
    st.error(f"⚠️ Error loading RAG chain: {e}")
    st.info("💡 Make sure you have installed all requirements and your vector DB is populated.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input handler
if query := st.chat_input("Ask about a book, author, or topic..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("🔍 Searching library archives..."):
        try:
            result = rag_chain.invoke({"input": query})
            answer = result["answer"]
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            with st.chat_message("assistant"):
                st.write(answer)
                
                # APA Citations
                st.markdown("---")
                st.markdown("<small>📎 <b>References</b></small>", unsafe_allow_html=True)
                seen_citations = set()
                
                if "context" in result and result["context"]:
                    for doc in result["context"]:
                        citation = build_apa_citation(doc.metadata)
                        if citation not in seen_citations:
                            seen_citations.add(citation)
                            st.caption(citation)
                else:
                    st.caption("ℹ️ No specific sources found for this answer.")

                # Source Chunks Expander
                with st.expander("🔎 View source chunks"):
                    for i, doc in enumerate(result.get("context", []), 1):
                        st.caption(f"Chunk {i} - Source: {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content)
                        st.divider()
                        
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
